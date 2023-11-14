/**
 * Created by user on 1/22/15.
 */

$(document).ready(function(){
    hide_elements();
});

function validate(form){
    return true;
}

function hide_element(elem){
    elem.parentElement.parentElement.hidden = true;
}

function hide_elements(){
    $('[id^=value_column_]').each(function(idx, element){
        if(idx == 0)
            return;
        hide_element(element);
    });
}

function unhide_elem(elem){
    elem.parentElement.parentElement.hidden = false;
}

function handle_value_column_click(caller, elem){
    if(caller.value === ''){
        hide_element(elem);
    }else{
        unhide_elem(elem);
    }
}