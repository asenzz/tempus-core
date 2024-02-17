/**
 * Created by user on 1/16/15.
 */

var rpc = new JsonRPC('/web/queue/ajax', ['getAllInputQueues']);
var gridId = "inputQueues";

$(document).ready(function () {
    initGrid();
    loadData();
});

function loadData() {
    $("#" + gridId)[0].grid.beginReq();
    rpc.getAllInputQueues.on_error = function (e) {
        alert(e.error);
    };
    rpc.getAllInputQueues.on_result = function (r) {

        // set the new data
        $("#" + gridId).jqGrid('setGridParam', {data: r});
        // hide the show message
        $("#" + gridId)[0].grid.endReq();
        // refresh the grid
        $("#" + gridId).trigger('reloadGrid');
    };

    rpc.getAllInputQueues();
}

function initGrid() {
    $("#" + gridId).jqGrid({
        datatype: "local",
        colModel: [
            { label: 'Table name', name: 'table_name', key: true, width: 75, formatter: function(cell, row){
                return '<a href="' + document.URL + 'show/' + cell + '">' + cell + '</a>';
            }},
            { label: 'Logical name', name: 'queue_name', width: 75 },
            { label: 'User', name: 'owner', width: 75 },
            { label: 'Description', name: 'description', width: 75 },
            { label: 'Resolution', name: 'resolution', width: 75 },
            { label: 'Legal Time Deviation', name: 'legal_time_deviation', width: 75 },
            { label: 'Columns', name: 'data_columns', width: 75 },
            { label: 'Time zone', name: 'timezone', width: 75 },
            { label: 'Is sparse?', name: 'is_sparse', width: 75, formatter:function(cell){
                return '<input type="checkbox" ' + (cell == true ? 'checked' : '') + ' disabled></input>';
            }}

        ],
        viewrecords: true,
        width: 800,
        height: 250,
        rowNum: 20,
        pager: "#jqGridPager"
    });
}