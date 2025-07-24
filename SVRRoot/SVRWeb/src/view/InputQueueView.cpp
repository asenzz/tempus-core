#include "view/InputQueueView.hpp"


namespace content
{

InputQueueForm::InputQueueForm()
{
    logical_name.id("logical_name");
    logical_name.name("logical_name");
    logical_name.message("Queue Name:");
    logical_name.regex(booster::regex(logical_name_pattern));
    logical_name.error_message("* The name must start with letter and can contain only letters, numbers or _");
    logical_name.non_empty();
    add(logical_name);

    description.id("description");
    description.name("description");
    description.message("Description:");
    add(description);

    resolution.id("resolution");
    resolution.name("resolution");
    resolution.message("Resolution:");
    resolution.help("Accepted format is " + time_duration_pattern);
    resolution.non_empty();
    add(resolution);

    legal_time_deviation.id("legal_time_deviation");
    legal_time_deviation.name("legal_time_deviation");
    legal_time_deviation.message("Legal Time Deviation:");
    legal_time_deviation.help("Accepted format is " + time_duration_pattern);
    legal_time_deviation.non_empty();
    add(legal_time_deviation);

    timezone.id("timezone");
    timezone.name("timezone");
    timezone.message("Timezone:");

    add(timezone);

    size_t input_fields = 10;

    for (size_t i = 1; i < input_fields; i++) {
        value_columns.push_back(std::make_unique<cppcms::widgets::text>());

        std::string id_prefix = "value_column_";
        std::string id = id_prefix + std::to_string(i);

        value_columns[i - 1]->id(id);
        value_columns[i - 1]->name(id);
        value_columns[i - 1]->message("Value column " + std::to_string(i) + ":");

        if (i + 1 < input_fields) {
            value_columns[i - 1]->attributes_string("onkeyup=\"handle_value_column_click(this, " + id_prefix + std::to_string(
                    i + 1) + ");\"");
        }

        add(*value_columns.back());
    }

    save.id("submit");
    save.name("submit");
    save.message("Save");
    save.value("Save");
    add(save);

}

bool InputQueueForm::validate()
{

    bool result = form::validate()
                  && validate_resolution()
                  && validate_legal_time_deviation()
                  && validate_value_columns();

    return result;
}

bool InputQueueForm::validate_resolution()
{
    LOG4_DEBUG("Validating resolution");

    try {
        bpt::time_duration _resolution = bpt::duration_from_string(resolution.value());

        if (_resolution < bpt::seconds(1) || _resolution > bpt::hours(24)) {
            throw std::logic_error("* Resolution must be more than 1 second and less than or equal to 24 hours.");
        }
    } catch (const std::exception &e) {
        resolution.valid(false);
        resolution.error_message(e.what());
        return false;
    }

    return true;
}

bool InputQueueForm::validate_legal_time_deviation()
{
    LOG4_DEBUG("Validating legal time deviation");

    try {
        bpt::time_duration _legal_time_dev = bpt::duration_from_string(legal_time_deviation.value());

        if (_legal_time_dev < bpt::seconds(0) || _legal_time_dev > bpt::hours(24)) {
            throw std::logic_error("* Legal time deviation must be positive and less than or equal to 24 hours.");
        }
    } catch (const std::exception &e) {
        legal_time_deviation.valid(false);
        legal_time_deviation.error_message(e.what());
        return false;
    }

    return true;
}

bool InputQueueForm::validate_value_columns()
{
    LOG4_DEBUG("Validating Queue value column names");
    std::vector<std::string> v_columns;

    for (auto & value_column_field : value_columns) {
        if (value_column_field->value().size() > 0) {
            v_columns.push_back(value_column_field->value());
        }
    }

    if (v_columns.size() == 0) {
        value_columns[0]->valid(false);
        value_columns[0]->error_message("* At least one (non-empty) column must be specified");
        return false;
    }

    return true;
}

}
