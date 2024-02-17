#pragma once
#include <model/RowsFrame.hpp>


namespace svr{
namespace datamodel{

class FramesContainer
{
private:
    const size_t frame_size;
    std::vector<RowsFrame_ptr> frames;
    bpt::time_duration resolution;
    std::vector<std::string> value_columns;

public:

    FramesContainer(size_t frame_size, const bpt::time_duration &resolution, const std::vector<std::string> &value_columns):
            frame_size(frame_size),
            resolution(resolution),
            value_columns(value_columns){}

    size_t get_frame_size() const {
        return frame_size;
    }

    std::vector<RowsFrame_ptr> const &get_frames() const {
        return frames;
    }

    void set_frames(std::vector<RowsFrame_ptr> const &frames_) {
        FramesContainer::frames = frames_;
    }

    size_t count_rows() const {
        size_t total = 0;
        for(const auto &f : get_frames()){
            total += f->size();
        }

        return total;
    }

    void add_frame(RowsFrame_ptr frame){
        if(frame->get_resolution() != resolution
                || frame->get_value_columns() != value_columns
                || frame->get_frame_size() != frame_size){
            LOG4_ERROR("Incompatible frame being added to container! " << frame->to_string());
            throw common::invalid_data("Invalid RowsFrame: " + frame->to_string());
        }
        frames.push_back(frame);
    }
};
}
}

using FramesContainer_ptr = std::shared_ptr<svr::datamodel::FramesContainer>;
