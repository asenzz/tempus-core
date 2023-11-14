/**
 * Created by user on 1/16/15.
 */

var rpc = new JsonRPC('/web/dataset/ajax', ['getAllDatasets']);
var gridId = "datasets";

$(document).ready(function () {
    initGrid();
    loadData();
});

function loadData() {
    $("#" + gridId)[0].grid.beginReq();
    rpc.getAllDatasets.on_error = function (e) {
        alert(e.error)
    };
    rpc.getAllDatasets.on_result = function (r) {

        // set the new data
        $("#" + gridId).jqGrid('setGridParam', {data: r});
        // hide the show message
        $("#" + gridId)[0].grid.endReq();
        // refresh the grid
        $("#" + gridId).trigger('reloadGrid');
    };

    rpc.getAllDatasets();
}

function initGrid() {
    $("#" + gridId).jqGrid({
        datatype: "local",
        colModel: [
            {label: 'Dataset ID', name: 'dataset_id', hidden:true},
            {label: 'Dataset Name', name: 'dataset_name', key: true, width: 75, formatter: function(cell, row){
                return '<a href="' + document.URL + 'show/' + cell + '">' + cell + '</a>';
            }},
            { label: 'User', name: 'user_name', width: 75 },
            { label: 'Priority', name: 'priority', width: 75 },
            { label: 'Description', name: 'description', width: 75 },
            { label: 'SWT Levels', name: 'swt_levels', width: 75 },
            { label: 'SWT Wavelet', name: 'swt_wavelet_name', width: 75 },
            { label: 'Lookback Time', name: 'lookback_time', width: 75 },
            { label: 'C coef.', name: 'svr_c', width: 35 },
            { label: 'Epsilon', name: 'svr_epsilon', width: 35 },
            { label: 'Kernel Param', name: 'svr_kernel_param', width: 35 },
            { label: 'Kernel Param 2', name: 'svr_kernel_param2', width: 35 },
            { label: 'Decr. Dist.', name: 'svr_decremental_distance', width: 35 },
            { label: 'Adj. Levels Ratio', name: 'svr_adjacent_levels_ratio', width: 35 },
            { label: 'Is Sparse', name: 'is_sparse', width: 35 }
        ],
        viewrecords: true,
        width: 700,
        height: 250,
        rowNum: 20,
        pager: "#" + gridId + 'Pager'
    });

    $('#' + gridId).navGrid('#' + gridId + 'Pager', {refresh:false, edit:false, add:false, del:false, search:false})
        .navButtonAdd('#' + gridId + 'Pager', {
            caption: "Add",
            buttonicon: "ui-icon-add",
            onClickButton : function(){
                window.location = window.location + 'create';
            }
        })
}
