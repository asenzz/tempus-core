/**
 * Created by user on 1/16/15.
 */

var rpc = new JsonRPC('/web/queue/ajax', ['showInputQueue', 'getValueColumnsModel']);
var gridId = "queue";
var colModel = [];


$(document).ready(function () {
    if($("#pageError") == undefined || $("#pageError").value == undefined)
        getValueColumnsModel();
});

function getValueColumnsModel(){
    rpc.getValueColumnsModel.on_error = function(e){
        show_alert("Warning", e);
    };

    rpc.getValueColumnsModel.on_result = function(r){
        colModel = [];

        colModel.push({ label: 'Value Time', name: 'value_time', key: true, width: 180});
        colModel.push({ label: 'Update Time', name: 'update_time', width: 180 });
        colModel.push({ label: 'Weight', name: 'weight', width: 35, align:"right"});
        colModel.push({ label: 'Is Final?', name: 'is_final', width: 35, formatter: function(cell, row){
            return '<input type="checkbox" ' + (cell == true ? 'checked' : '') + ' disabled></input>';
        }});

        for(var c = 0; c < r.length; c++){
            colModel.push(r[c]);
        }
        initGrid();
    };

    rpc.getValueColumnsModel(inputQueueTableName);
}

function loadData() {
    $("#" + gridId)[0].grid.beginReq();
    rpc.showInputQueue.on_error = function (e) {
        alert(e.error)
    };
    rpc.showInputQueue.on_result = function (r) {

        // set the new data
        $("#" + gridId).jqGrid('setGridParam', {data: r});
        // hide the show message
        $("#" + gridId)[0].grid.endReq();
        // refresh the grid
        $("#" + gridId).trigger('reloadGrid');
    };

    rpc.showInputQueue(inputQueueTableName);
}

function initGrid() {

    $("#" + gridId).jqGrid({
        datatype: "local",
        colModel: colModel,
        viewrecords: true,
        width: 700,
        height: 250,
        rowNum: 20,
        sortname: 'value_time',
        sortorder: 'desc',
        pager: "#jqGridPager"
    });
    loadData();
}
