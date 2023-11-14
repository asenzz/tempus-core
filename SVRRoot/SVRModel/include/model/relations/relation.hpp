#ifndef RELATION_HPP
#define RELATION_HPP

#include <memory>
#include <string>

namespace svr {
namespace datamodel{

template <class related_obj_t, class storage_connector_t, typename database_id_t = uint64_t>
class relation
{
public:
    using related_object_ptr_t = std::shared_ptr<related_obj_t>;

    relation()=default;
    relation(database_id_t);
    relation(related_object_ptr_t);
    relation & operator=(relation const & other);

    database_id_t get_id() const;
    void set_id(database_id_t);
    related_object_ptr_t get_obj();
    void set_obj(related_object_ptr_t);
private:
    database_id_t id;
    related_object_ptr_t ptr;

    void ensure_id();
};


/******************************************************************************/


template <class related_obj_t, class storage_connector_t, typename database_id_t>
relation<related_obj_t, storage_connector_t, database_id_t>::relation(database_id_t id)
: id(id)
, ptr(nullptr)
{}


template <class related_obj_t, class storage_connector_t, typename database_id_t>
relation<related_obj_t, storage_connector_t, database_id_t>::relation(related_object_ptr_t ptr)
: ptr(ptr)
{
    ensure_id();
}


template <class related_obj_t, class storage_connector_t, typename database_id_t>
relation<related_obj_t, storage_connector_t, database_id_t> & relation<related_obj_t, storage_connector_t, database_id_t>::operator=(relation const & other)
{
    id = other.id;
    ptr = other.ptr;
    return *this;
}


template <class related_obj_t, class storage_connector_t, typename database_id_t>
database_id_t relation<related_obj_t, storage_connector_t, database_id_t>::get_id() const
{
    return id;
}


template <class related_obj_t, class storage_connector_t, typename database_id_t>
void relation<related_obj_t, storage_connector_t, database_id_t>::set_id(database_id_t id_)
{
    id = id_;
    ptr = nullptr;
}


template <class related_obj_t, class storage_connector_t, typename database_id_t>
typename relation<related_obj_t, storage_connector_t, database_id_t>::related_object_ptr_t relation<related_obj_t, storage_connector_t, database_id_t>::get_obj()
{
    if(!ptr)
        ptr = storage_connector_t::load(id);
    return ptr;
}


template <class related_obj_t, class storage_connector_t, typename database_id_t>
void relation<related_obj_t, storage_connector_t, database_id_t>::set_obj(related_object_ptr_t ptr_)
{
    ptr = ptr_;
    ensure_id();
}

template <class related_obj_t, class storage_connector_t, typename database_id_t>
void relation<related_obj_t, storage_connector_t, database_id_t>::ensure_id()
{
    if(ptr->get_id() == 0)
        ptr->set_id( storage_connector_t::get_next_id() );
}


/******************************************************************************/
/******************************************************************************/
/******************************************************************************/


template <class related_obj_t, class storage_connector_t>
class relation <related_obj_t, storage_connector_t, std::string>
{
public:
    using related_object_ptr_t = std::shared_ptr<related_obj_t>;

    relation()=default;
    relation(std::string const &);
    relation(related_object_ptr_t);

    std::string const & get_id() const;
    void set_id(std::string const &);

    related_object_ptr_t const & get_obj() const;
    void set_obj(related_object_ptr_t);
private:
    std::string id;
    mutable related_object_ptr_t ptr;
};


/******************************************************************************/


template <class related_obj_t, class storage_connector_t>
relation<related_obj_t, storage_connector_t, std::string>::relation(std::string const & id)
: id(id)
, ptr(nullptr)
{}


template <class related_obj_t, class storage_connector_t>
relation<related_obj_t, storage_connector_t, std::string>::relation(related_object_ptr_t ptr)
: ptr(ptr)
{
}


template <class related_obj_t, class storage_connector_t>
std::string const & relation<related_obj_t, storage_connector_t, std::string>::get_id() const
{
    return id;
}


template <class related_obj_t, class storage_connector_t>
void relation<related_obj_t, storage_connector_t, std::string>::set_id(std::string const & id_)
{
    id = id_;
    ptr = nullptr;
}


template <class related_obj_t, class storage_connector_t>
typename relation<related_obj_t, storage_connector_t, std::string>::related_object_ptr_t const & relation<related_obj_t, storage_connector_t, std::string>::get_obj() const
{
    if(!ptr)
        ptr = storage_connector_t::load(id);
    return ptr;
}


template <class related_obj_t, class storage_connector_t>
void relation<related_obj_t, storage_connector_t, std::string>::set_obj(related_object_ptr_t ptr_)
{
    ptr = ptr_;
}


}
}


#endif /* RELATION_HPP */
