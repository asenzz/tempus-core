#
# Removes the specified compile flag from the specified target.
#   _target     - The target to remove the compile flag from
#   _flag       - The compile flag to remove
#
# Pre: apply_global_cxx_flags_to_all_targets() must be invoked.
#
macro(remove_compile_flag_from_target _target _flag)
    get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
    if(_target_cxx_flags)
        list(REMOVE_ITEM _target_cxx_flags ${_flag})
        set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
    endif()
endmacro()

macro(remove_link_flag_from_target _target _flag)
    get_target_property(_target_cxx_flags ${_target} LINK_OPTIONS)
    if(_target_cxx_flags)
        list(REMOVE_ITEM _target_cxx_flags ${_flag})
        set_target_properties(${_target} PROPERTIES LINK_OPTIONS "${_target_cxx_flags}")
    endif()
endmacro()

macro(remove_compile_flag_from_dir _flag)
    get_directory_property(_dir_compile_flags COMPILE_OPTIONS)
    if(_dir_compile_flags)
        list(REMOVE_ITEM _dir_compile_flags ${_flag})
        set_directory_properties(PROPERTIES COMPILE_OPTIONS "${_dir_compile_flags}")
    endif()
endmacro()

macro(remove_link_flag_from_dir _flag)
    get_directory_property(_dir_link_flags LINK_OPTIONS)
    if(_dir_link_flags)
        list(REMOVE_ITEM _dir_link_flags ${_flag})
        set_directory_properties(PROPERTIES LINK_OPTIONS "${_dir_link_flags}")
    endif()
endmacro()
