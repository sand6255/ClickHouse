#include "PyTorchModel.h"

#include <Common/FieldVisitorConvertToNumber.h>
#include <mutex>
#include <Columns/ColumnString.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnTuple.h>
#include <Common/typeid_cast.h>
#include <IO/WriteBufferFromString.h>
#include <IO/Operators.h>
#include <Common/PODArray.h>
#include <Common/SharedLibrary.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeTuple.h>
#include <torch/script.h>


namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int BAD_ARGUMENTS;
extern const int CANNOT_LOAD_PYTORCH_MODEL;
extern const int CANNOT_APPLY_PYTORCH_MODEL;
}

namespace
{

class PyTorchModelImpl : public IPyTorchModel
{
public:
    PyTorchModelImpl(const std::string & model_path)
    {
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load(model_path);
        }
        catch (const c10::Error& e) {
            std::string msg = "Cannot load PyTorch model: ";
            throw Exception(msg + e.msg(), ErrorCodes::CANNOT_LOAD_PYTORCH_MODEL);
        }

        //float_features_count = api->GetFloatFeaturesCount(handle->get());
        //cat_features_count = api->GetCatFeaturesCount(handle->get());
        //if (api->GetDimensionsCount)
        //    tree_count = api->GetDimensionsCount(handle->get());
    }

    ColumnPtr evaluate(const ColumnRawPtrs & columns) override
    {
        if (columns.empty())
            throw Exception("Got empty columns list for PyTorch model.", ErrorCodes::BAD_ARGUMENTS);

        auto result = evalImpl(columns);

        return result;
    }

//    size_t getFloatFeaturesCount() const override { return float_features_count; }
//    size_t getCatFeaturesCount() const override { return cat_features_count; }
//    size_t getTreeCount() const override { return tree_count; }

private:
//    size_t float_features_count;
//    size_t cat_features_count;
//    size_t tree_count;
    torch::jit::script::Module module;

    /// Buffer should be allocated with features_count * column->size() elements.
    /// Place column elements in positions buffer[0], buffer[features_count], ... , buffer[size * features_count]
/*    template <typename T>
    void placeColumnAsNumber(const IColumn * column, T * buffer, size_t features_count) const
    {
        size_t size = column->size();
        FieldVisitorConvertToNumber<T> visitor;
        for (size_t i = 0; i < size; ++i)
        {
            /// TODO: Replace with column visitor.
            Field field;
            column->get(i, field);
            *buffer = applyVisitor(visitor, field);
            buffer += features_count;
        }
    }

    /// Buffer should be allocated with features_count * column->size() elements.
    /// Place string pointers in positions buffer[0], buffer[features_count], ... , buffer[size * features_count]
    static void placeStringColumn(const ColumnString & column, const char ** buffer, size_t features_count)
    {
        size_t size = column.size();
        for (size_t i = 0; i < size; ++i)
        {
            *buffer = const_cast<char *>(column.getDataAtWithTerminatingZero(i).data);
            buffer += features_count;
        }
    }

    /// Buffer should be allocated with features_count * column->size() elements.
    /// Place string pointers in positions buffer[0], buffer[features_count], ... , buffer[size * features_count]
    /// Returns PODArray which holds data (because ColumnFixedString doesn't store terminating zero).
    static PODArray<char> placeFixedStringColumn(
            const ColumnFixedString & column, const char ** buffer, size_t features_count)
    {
        size_t size = column.size();
        size_t str_size = column.getN();
        PODArray<char> data(size * (str_size + 1));
        char * data_ptr = data.data();

        for (size_t i = 0; i < size; ++i)
        {
            auto ref = column.getDataAt(i);
            memcpy(data_ptr, ref.data, ref.size);
            data_ptr[ref.size] = 0;
            *buffer = data_ptr;
            data_ptr += ref.size + 1;
            buffer += features_count;
        }

        return data;
    }

    /// Place columns into buffer, returns column which holds placed data. Buffer should contains column->size() values.
    template <typename T>
    ColumnPtr placeNumericColumns(const ColumnRawPtrs & columns,
                                  size_t offset, size_t size, const T** buffer) const
    {
        if (size == 0)
            return nullptr;
        size_t column_size = columns[offset]->size();
        auto data_column = ColumnVector<T>::create(size * column_size);
        T * data = data_column->getData().data();
        for (size_t i = 0; i < size; ++i)
        {
            const auto * column = columns[offset + i];
            if (column->isNumeric())
                placeColumnAsNumber(column, data + i, size);
        }

        for (size_t i = 0; i < column_size; ++i)
        {
            *buffer = data;
            ++buffer;
            data += size;
        }

        return data_column;
    }

    /// Place columns into buffer, returns data which was used for fixed string columns.
    /// Buffer should contains column->size() values, each value contains size strings.
    static std::vector<PODArray<char>> placeStringColumns(
            const ColumnRawPtrs & columns, size_t offset, size_t size, const char ** buffer)
    {
        if (size == 0)
            return {};

        std::vector<PODArray<char>> data;
        for (size_t i = 0; i < size; ++i)
        {
            const auto * column = columns[offset + i];
            if (const auto * column_string = typeid_cast<const ColumnString *>(column))
                placeStringColumn(*column_string, buffer + i, size);
            else if (const auto * column_fixed_string = typeid_cast<const ColumnFixedString *>(column))
                data.push_back(placeFixedStringColumn(*column_fixed_string, buffer + i, size));
            else
                throw Exception("Cannot place string column.", ErrorCodes::LOGICAL_ERROR);
        }

        return data;
    }

    /// Calc hash for string cat feature at ps positions.
    template <typename Column>
    void calcStringHashes(const Column * column, size_t ps, const int ** buffer) const
    {
        size_t column_size = column->size();
        for (size_t j = 0; j < column_size; ++j)
        {
            auto ref = column->getDataAt(j);
            //const_cast<int *>(*buffer)[ps] = api->GetStringCatFeatureHash(ref.data, ref.size);
            ++buffer;
        }
    }

    /// Calc hash for int cat feature at ps position. Buffer at positions ps should contains unhashed values.
    void calcIntHashes(size_t column_size, size_t ps, const int ** buffer) const
    {
        for (size_t j = 0; j < column_size; ++j)
        {
            //const_cast<int *>(*buffer)[ps] = api->GetIntegerCatFeatureHash((*buffer)[ps]);
            ++buffer;
        }
    }

    /// buffer contains column->size() rows and size columns.
    /// For int cat features calc hash inplace.
    /// For string cat features calc hash from column rows.
    void calcHashes(const ColumnRawPtrs & columns, size_t offset, size_t size, const int ** buffer) const
    {
        if (size == 0)
            return;
        size_t column_size = columns[offset]->size();

        std::vector<PODArray<char>> data;
        for (size_t i = 0; i < size; ++i)
        {
            const auto * column = columns[offset + i];
            if (const auto * column_string = typeid_cast<const ColumnString *>(column))
                calcStringHashes(column_string, i, buffer);
            else if (const auto * column_fixed_string = typeid_cast<const ColumnFixedString *>(column))
                calcStringHashes(column_fixed_string, i, buffer);
            else
                calcIntHashes(column_size, i, buffer);
        }
    }

    /// buffer[column_size * cat_features_count] -> char * => cat_features[column_size][cat_features_count] -> char *
    void fillCatFeaturesBuffer(const char *** cat_features, const char ** buffer,
                               size_t column_size) const
    {
        for (size_t i = 0; i < column_size; ++i)
        {
            *cat_features = buffer;
            ++cat_features;
            buffer += cat_features_count;
        }
    }
*/
    /// Convert values to row-oriented format and call evaluation function from PyTorch wrapper api.
    ///  * CalcModelPredictionFlat if no cat features
    ///  * CalcModelPrediction if all cat features are strings
    ///  * CalcModelPredictionWithHashedCatFeatures if has int cat features.
    ColumnFloat64::MutablePtr evalImpl(const ColumnRawPtrs & columns)
    {
        std::string error_msg = "Error occurred while applying PyTorch model: ";
        size_t column_size = columns.front()->size();

        auto result = ColumnFloat64::create(column_size);

        std::vector<torch::jit::IValue> inputs;

        inputs.push_back(torch::ones({1, 3, 224, 224}));
        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

        return result;
    }
};

}


PyTorchModel::PyTorchModel(std::string name_, std::string model_path_,
                             const ExternalLoadableLifetime & lifetime_)
    : name(std::move(name_)), model_path(std::move(model_path_)), lifetime(lifetime_)
{
    model = std::make_unique<PyTorchModelImpl>(model_path);
}

const ExternalLoadableLifetime & PyTorchModel::getLifetime() const
{
    return lifetime;
}

bool PyTorchModel::isModified() const
{
    return true;
}

std::shared_ptr<const IExternalLoadable> PyTorchModel::clone() const
{
    return std::make_shared<PyTorchModel>(name, model_path, lifetime);
}

/*size_t PyTorchModel::getFloatFeaturesCount() const
{
    return float_features_count;
}

size_t PyTorchModel::getCatFeaturesCount() const
{
    return cat_features_count;
}

size_t PyTorchModel::getTreeCount() const
{
    return tree_count;
}*/

DataTypePtr PyTorchModel::getReturnType() const
{
    auto type = std::make_shared<DataTypeFloat64>();
    //if (tree_count == 1)
        return type;

    //DataTypes types(tree_count, type);

    //return std::make_shared<DataTypeTuple>(types);
}

ColumnPtr PyTorchModel::evaluate(const ColumnRawPtrs & columns) const
{
    if (!model)
        throw Exception("PyTorch model was not loaded.", ErrorCodes::LOGICAL_ERROR);
    return model->evaluate(columns);
}

}
