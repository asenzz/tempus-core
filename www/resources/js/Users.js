/**
 * Created by user on 1/16/15.
 */

var rpc = new JsonRPC('/web/user/ajax', ['getAllUsers', 'createUser']);
var gridId = "users";

$(document).ready(function () {
    initGrid();
    loadData();
});

function loadData() {
    $("#" + gridId)[0].grid.beginReq();
    rpc.getAllUsers.on_error = function (e) {
        alert(e.error)
    };
    rpc.getAllUsers.on_result = function (r) {

        // set the new data
        $("#" + gridId).jqGrid('setGridParam', {data: r});
        // hide the show message
        $("#" + gridId)[0].grid.endReq();
        // refresh the grid
        $("#" + gridId).trigger('reloadGrid');
    };

    rpc.getAllUsers();
}

function initGrid() {
    $("#" + gridId).jqGrid({
        datatype: "local",
        colModel: [
            {label: 'user_name', name: 'user_name', key: true, width: 75, formatter: function(cell, row){
                return '<a href="' + document.URL + 'show/' + cell + '">' + cell + '</a>';
            }},
            { label: 'id', name: 'id', width: 75 },
            { label: 'email', name: 'email', width: 75 },
            { label: 'name', name: 'name', width: 75 }
        ],
        viewrecords: true,
        width: 700,
        height: 250,
        rowNum: 20,
        pager: "#jqGridPager"
    });
}
