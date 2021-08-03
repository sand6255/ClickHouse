#pragma once
#include <Interpreters/IExternalLoadable.h>
#include <Columns/IColumn.h>
#include <Columns/ColumnsNumber.h>
#include <Interpreters/CatBoostModel.h> // to change

namespace DB
{

/// PyTorch model interface.
class IPyTorchModel
{
public:
    virtual ~IPyTorchModel() = default;
    /// Evaluate model. Use first `float_features_count` columns as float features,
    /// the others `cat_features_count` as categorical features.
    virtual ColumnPtr evaluate(const ColumnRawPtrs & columns) = 0;

    //virtual size_t getFloatFeaturesCount() const = 0;
    //virtual size_t getCatFeaturesCount() const = 0;
    //virtual size_t getTreeCount() const = 0;
};

class PyTorchModel : public IModel
{
public:
    PyTorchModel(std::string name, std::string model_path,
                  const ExternalLoadableLifetime & lifetime);

    ColumnPtr evaluate(const ColumnRawPtrs & columns) const override;
    std::string getTypeName() const override { return "pytorch"; }

//    size_t getFloatFeaturesCount() const;
//    size_t getCatFeaturesCount() const;
//    size_t getTreeCount() const;
    DataTypePtr getReturnType() const override;

    /// IExternalLoadable interface.

    const ExternalLoadableLifetime & getLifetime() const override;

    const std::string & getLoadableName() const override { return name; }

    bool supportUpdates() const override { return true; }

    bool isModified() const override;

    std::shared_ptr<const IExternalLoadable> clone() const override;

private:
    const std::string name;
    std::string model_path;
    ExternalLoadableLifetime lifetime;

    std::unique_ptr<IPyTorchModel> model;

//    size_t float_features_count;
//    size_t cat_features_count;
//    size_t tree_count;

    void init();
};

}
