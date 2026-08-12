#include <map>
#include <string>

using Params = std::map<std::string, std::string>;

void configure_lightgbm(Params& params) {
    params["objective"] = "regression";
    params["metric"] = "mae";
    params["num_leaves"] = 50;
    params["learning_rate"] = 0.03;
    params["feature_fraction"] = 0.8;
    params["bagging_fraction"] = 0.8;
    params["boosting_type"] = "dart";
    params["drop_rate"] = 0.1;
}
