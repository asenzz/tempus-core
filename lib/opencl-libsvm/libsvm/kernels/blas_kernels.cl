

__kernel void dot_product(
                        __global const double16 * a,
                        __global const double16 * b,
                        __global double16 * result 
                        )
{
    uint const gid = get_global_id(0)
        , gsz = get_global_size(0)
        , lid = get_local_id(0)
        , lsz = get_local_size(0)
        , grid = get_group_id(0)
        , grcount = get_num_groups(0)
    ;
    
    result[gid] = a[gid] * b[gid];
    
//    printf("gid: %d, gsz: %d, lid: %d, lsz: %d, grid: %d, grcount: %d\n", gid, gsz, lid, lsz, grid, grcount);
}
