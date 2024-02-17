//
// Created by zarko on 2/12/24.
//

#include "onlinesvr_persist.tpp"

namespace svr {

OnlineMIMOSVR::OnlineMIMOSVR(std::stringstream &input_stream)
{
    boost::archive::binary_iarchive ia(input_stream);
    ia >> *this;
}

}