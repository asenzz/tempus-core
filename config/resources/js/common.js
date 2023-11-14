/**
 * Created by user on 1/16/15.
 */
$(document).ready(function(){
    // simple jMenu plugin called
    $("#jMenu").jMenu();

    // more complex jMenu plugin called
    $("#jMenu").jMenu({
        ulWidth : 'auto',
        effects : {
            effectSpeedOpen : 300,
            effectTypeClose : 'slide'
        },
        animatedText : false
    });
});

function show_alert(title, body){
    $("#jDialog").innerHTML = body;
    $("#jDialog").dialog({
            title: title
        }
    ).open();
}