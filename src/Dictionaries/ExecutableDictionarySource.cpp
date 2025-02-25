#include "ExecutableDictionarySource.h"

#include <functional>
#include <common/scope_guard.h>
#include <DataStreams/OwningBlockInputStream.h>
#include <DataStreams/formatBlock.h>
#include <Processors/ISimpleTransform.h>
#include <Processors/QueryPipeline.h>
#include <Processors/Executors/PullingPipelineExecutor.h>
#include <Processors/Sources/SourceWithProgress.h>
#include <Processors/Formats/IInputFormat.h>
#include <Interpreters/Context.h>
#include <Formats/FormatFactory.h>
#include <IO/WriteHelpers.h>
#include <IO/ReadHelpers.h>
#include <Common/ShellCommand.h>
#include <Common/ThreadPool.h>
#include <common/logger_useful.h>
#include <common/LocalDateTime.h>
#include "DictionarySourceFactory.h"
#include "DictionarySourceHelpers.h"
#include "DictionaryStructure.h"
#include "registerDictionaries.h"


namespace DB
{

static const UInt64 max_block_size = 8192;

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int DICTIONARY_ACCESS_DENIED;
    extern const int UNSUPPORTED_METHOD;
}

namespace
{
    /// Owns ShellCommand and calls wait for it.
    class ShellCommandOwningTransform final : public ISimpleTransform
    {
    private:
        Poco::Logger * log;
        std::unique_ptr<ShellCommand> command;
    public:
        ShellCommandOwningTransform(const Block & header, Poco::Logger * log_, std::unique_ptr<ShellCommand> command_)
            : ISimpleTransform(header, header, true), log(log_), command(std::move(command_))
        {
        }

        String getName() const override { return "ShellCommandOwningTransform"; }
        void transform(Chunk &) override {}

        Status prepare() override
        {
            auto status = ISimpleTransform::prepare();
            if (status == Status::Finished)
            {
                std::string err;
                readStringUntilEOF(err, command->err);
                if (!err.empty())
                    LOG_ERROR(log, "Having stderr: {}", err);

                command->wait();
            }

            return status;
        }
    };

}

ExecutableDictionarySource::ExecutableDictionarySource(
    const DictionaryStructure & dict_struct_,
    const Configuration & configuration_,
    Block & sample_block_,
    ContextPtr context_)
    : log(&Poco::Logger::get("ExecutableDictionarySource"))
    , dict_struct(dict_struct_)
    , configuration(configuration_)
    , sample_block{sample_block_}
    , context(context_)
{
    /// Remove keys from sample_block for implicit_key dictionary because
    /// these columns will not be returned from source
    if (configuration.implicit_key)
    {
        auto keys_names = dict_struct.getKeysNames();

        for (auto & key_name : keys_names)
        {
            size_t key_column_position_in_block = sample_block.getPositionByName(key_name);
            sample_block.erase(key_column_position_in_block);
        }
    }
}

ExecutableDictionarySource::ExecutableDictionarySource(const ExecutableDictionarySource & other)
    : log(&Poco::Logger::get("ExecutableDictionarySource"))
    , update_time(other.update_time)
    , dict_struct(other.dict_struct)
    , configuration(other.configuration)
    , sample_block(other.sample_block)
    , context(Context::createCopy(other.context))
{
}

Pipe ExecutableDictionarySource::loadAll()
{
    if (configuration.implicit_key)
        throw Exception(ErrorCodes::UNSUPPORTED_METHOD, "ExecutableDictionarySource with implicit_key does not support loadAll method");

    LOG_TRACE(log, "loadAll {}", toString());
    auto process = ShellCommand::execute(configuration.command);
    Pipe pipe(FormatFactory::instance().getInput(configuration.format, process->out, sample_block, context, max_block_size));
    pipe.addTransform(std::make_shared<ShellCommandOwningTransform>(pipe.getHeader(), log, std::move(process)));
    return pipe;
}

Pipe ExecutableDictionarySource::loadUpdatedAll()
{
    if (configuration.implicit_key)
        throw Exception(ErrorCodes::UNSUPPORTED_METHOD, "ExecutableDictionarySource with implicit_key does not support loadUpdatedAll method");

    time_t new_update_time = time(nullptr);
    SCOPE_EXIT(update_time = new_update_time);

    std::string command_with_update_field = configuration.command;
    if (update_time)
        command_with_update_field += " " + configuration.update_field + " " + DB::toString(LocalDateTime(update_time - configuration.update_lag));

    LOG_TRACE(log, "loadUpdatedAll {}", command_with_update_field);
    auto process = ShellCommand::execute(command_with_update_field);

    Pipe pipe(FormatFactory::instance().getInput(configuration.format, process->out, sample_block, context, max_block_size));
    pipe.addTransform(std::make_shared<ShellCommandOwningTransform>(pipe.getHeader(), log, std::move(process)));
    return pipe;
}

namespace
{
    /** A stream, that runs child process and sends data to its stdin in background thread,
      *  and receives data from its stdout.
      *
      *  TODO: implement without background thread.
      */
    class SourceWithBackgroundThread final : public SourceWithProgress
    {
    public:
        SourceWithBackgroundThread(
            ContextPtr context,
            const std::string & format,
            const Block & sample_block,
            const std::string & command_str,
            Poco::Logger * log_,
            std::function<void(WriteBufferFromFile &)> && send_data_)
            : SourceWithProgress(sample_block)
            , log(log_)
            , command(ShellCommand::execute(command_str))
            , send_data(std::move(send_data_))
            , thread([this] { send_data(command->in); })
        {
            pipeline.init(Pipe(FormatFactory::instance().getInput(format, command->out, sample_block, context, max_block_size)));
            executor = std::make_unique<PullingPipelineExecutor>(pipeline);
        }

        ~SourceWithBackgroundThread() override
        {
            if (thread.joinable())
                thread.join();
        }

    protected:
        Chunk generate() override
        {
            Chunk chunk;
            executor->pull(chunk);
            return chunk;
        }

    public:
        Status prepare() override
        {
            auto status = SourceWithProgress::prepare();

            if (status == Status::Finished)
            {
                std::string err;
                readStringUntilEOF(err, command->err);
                if (!err.empty())
                    LOG_ERROR(log, "Having stderr: {}", err);

                if (thread.joinable())
                    thread.join();

                command->wait();
            }

            return status;
        }

        String getName() const override { return "SourceWithBackgroundThread"; }

        Poco::Logger * log;
        QueryPipeline pipeline;
        std::unique_ptr<PullingPipelineExecutor> executor;
        std::unique_ptr<ShellCommand> command;
        std::function<void(WriteBufferFromFile &)> send_data;
        ThreadFromGlobalPool thread;
    };
}

Pipe ExecutableDictionarySource::loadIds(const std::vector<UInt64> & ids)
{
    LOG_TRACE(log, "loadIds {} size = {}", toString(), ids.size());

    auto block = blockForIds(dict_struct, ids);
    return getStreamForBlock(block);
}

Pipe ExecutableDictionarySource::loadKeys(const Columns & key_columns, const std::vector<size_t> & requested_rows)
{
    LOG_TRACE(log, "loadKeys {} size = {}", toString(), requested_rows.size());

    auto block = blockForKeys(dict_struct, key_columns, requested_rows);
    return getStreamForBlock(block);
}

Pipe ExecutableDictionarySource::getStreamForBlock(const Block & block)
{
    Pipe pipe(std::make_unique<SourceWithBackgroundThread>(
        context, configuration.format, sample_block, configuration.command, log,
        [block, this](WriteBufferFromFile & out) mutable
        {
            auto output_stream = context->getOutputStream(configuration.format, out, block.cloneEmpty());
            formatBlock(output_stream, block);
            out.close();
        }));

    if (configuration.implicit_key)
        pipe.addTransform(std::make_shared<TransformWithAdditionalColumns>(block, pipe.getHeader()));

    return pipe;
}

bool ExecutableDictionarySource::isModified() const
{
    return true;
}

bool ExecutableDictionarySource::supportsSelectiveLoad() const
{
    return true;
}

bool ExecutableDictionarySource::hasUpdateField() const
{
    return !configuration.update_field.empty();
}

DictionarySourcePtr ExecutableDictionarySource::clone() const
{
    return std::make_unique<ExecutableDictionarySource>(*this);
}

std::string ExecutableDictionarySource::toString() const
{
    return "Executable: " + configuration.command;
}

void registerDictionarySourceExecutable(DictionarySourceFactory & factory)
{
    auto create_table_source = [=](const DictionaryStructure & dict_struct,
                                 const Poco::Util::AbstractConfiguration & config,
                                 const std::string & config_prefix,
                                 Block & sample_block,
                                 ContextPtr context,
                                 const std::string & /* default_database */,
                                 bool created_from_ddl) -> DictionarySourcePtr
    {
        if (dict_struct.has_expressions)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Dictionary source of type `executable` does not support attribute expressions");

        /// Executable dictionaries may execute arbitrary commands.
        /// It's OK for dictionaries created by administrator from xml-file, but
        /// maybe dangerous for dictionaries created from DDL-queries.
        if (created_from_ddl && context->getApplicationType() != Context::ApplicationType::LOCAL)
            throw Exception(ErrorCodes::DICTIONARY_ACCESS_DENIED, "Dictionaries with executable dictionary source are not allowed to be created from DDL query");

        auto context_local_copy = copyContextAndApplySettings(config_prefix, context, config);

        std::string settings_config_prefix = config_prefix + ".executable";

        ExecutableDictionarySource::Configuration configuration
        {
            .command = config.getString(settings_config_prefix + ".command"),
            .format = config.getString(settings_config_prefix + ".format"),
            .update_field = config.getString(settings_config_prefix + ".update_field", ""),
            .update_lag = config.getUInt64(settings_config_prefix + ".update_lag", 1),
            .implicit_key = config.getBool(settings_config_prefix + ".implicit_key", false)
        };

        return std::make_unique<ExecutableDictionarySource>(dict_struct, configuration, sample_block, context_local_copy);
    };

    factory.registerSource("executable", create_table_source);
}

}
