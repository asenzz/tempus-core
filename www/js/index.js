Object.size = function(obj) {
		    var size = 0, key;
		    for (key in obj) {
		        if (obj.hasOwnProperty(key)) size++;
		    }
		    return size;
		};
		
						
		var LZW = {
		    compress: function (uncompressed) {
		        "use strict";
		        // Build the dictionary.
		        var i,
		            dictionary = {},
		            c,
		            wc,
		            w = "",
		            result = [],
		            dictSize = 256;
		        for (i = 0; i < 256; i += 1) {
		            dictionary[String.fromCharCode(i)] = i;
		        }
		 
		        for (i = 0; i < uncompressed.length; i += 1) {
		            c = uncompressed.charAt(i);
		            wc = w + c;
		            if (dictionary[wc]) {
		                w = wc;
		            } else {
		                result.push(dictionary[w]);
		                // Add wc to the dictionary.
		                dictionary[wc] = dictSize++;
		                w = String(c);
		            }
		        }
		 
		        // Output the code for w.
		        if (w !== "") {
		            result.push(dictionary[w]);
		        }
		        return result;
		    },
		 
		 
		    decompress: function (compressed) {
		        "use strict";
		        // Build the dictionary.
		        var i,
		            dictionary = [],
		            w,
		            result,
		            k,
		            entry = "",
		            dictSize = 256;
		        for (i = 0; i < 256; i += 1) {
		            dictionary[i] = String.fromCharCode(i);
		        }
		 
		        w = String.fromCharCode(compressed[0]);
		        result = w;
		        for (i = 1; i < compressed.length; i += 1) {
		            k = compressed[i];
		            if (dictionary[k]) {
		                entry = dictionary[k];
		            } else {
		                if (k === dictSize) {
		                    entry = w + w.charAt(0);
		                } else {
		                    return null;
		                }
		            }
		 
		            result += entry;
		 
		            // Add w+entry[0] to the dictionary.
		            dictionary[dictSize++] = w + entry.charAt(0);
		 
		            w = entry;
		        }
		        return result;
		    }
		};
			
			/*  
		===============================================================================
		Crc32 is a JavaScript function for computing the CRC32 of a string
		...............................................................................
		
		Version: 1.2 - 2006/11 - http://noteslog.com/category/javascript/
		
		-------------------------------------------------------------------------------
		Copyright (c) 2006 Andrea Ercolino      
		http://www.opensource.org/licenses/mit-license.php
		===============================================================================
		*/
		
		(function() {
		  var strTable = "00000000 77073096 EE0E612C 990951BA 076DC419 706AF48F E963A535 9E6495A3 0EDB8832 79DCB8A4 E0D5E91E 97D2D988 09B64C2B 7EB17CBD E7B82D07 90BF1D91 1DB71064 6AB020F2 F3B97148 84BE41DE 1ADAD47D 6DDDE4EB F4D4B551 83D385C7 136C9856 646BA8C0 FD62F97A 8A65C9EC 14015C4F 63066CD9 FA0F3D63 8D080DF5 3B6E20C8 4C69105E D56041E4 A2677172 3C03E4D1 4B04D447 D20D85FD A50AB56B 35B5A8FA 42B2986C DBBBC9D6 ACBCF940 32D86CE3 45DF5C75 DCD60DCF ABD13D59 26D930AC 51DE003A C8D75180 BFD06116 21B4F4B5 56B3C423 CFBA9599 B8BDA50F 2802B89E 5F058808 C60CD9B2 B10BE924 2F6F7C87 58684C11 C1611DAB B6662D3D 76DC4190 01DB7106 98D220BC EFD5102A 71B18589 06B6B51F 9FBFE4A5 E8B8D433 7807C9A2 0F00F934 9609A88E E10E9818 7F6A0DBB 086D3D2D 91646C97 E6635C01 6B6B51F4 1C6C6162 856530D8 F262004E 6C0695ED 1B01A57B 8208F4C1 F50FC457 65B0D9C6 12B7E950 8BBEB8EA FCB9887C 62DD1DDF 15DA2D49 8CD37CF3 FBD44C65 4DB26158 3AB551CE A3BC0074 D4BB30E2 4ADFA541 3DD895D7 A4D1C46D D3D6F4FB 4369E96A 346ED9FC AD678846 DA60B8D0 44042D73 33031DE5 AA0A4C5F DD0D7CC9 5005713C 270241AA BE0B1010 C90C2086 5768B525 206F85B3 B966D409 CE61E49F 5EDEF90E 29D9C998 B0D09822 C7D7A8B4 59B33D17 2EB40D81 B7BD5C3B C0BA6CAD EDB88320 9ABFB3B6 03B6E20C 74B1D29A EAD54739 9DD277AF 04DB2615 73DC1683 E3630B12 94643B84 0D6D6A3E 7A6A5AA8 E40ECF0B 9309FF9D 0A00AE27 7D079EB1 F00F9344 8708A3D2 1E01F268 6906C2FE F762575D 806567CB 196C3671 6E6B06E7 FED41B76 89D32BE0 10DA7A5A 67DD4ACC F9B9DF6F 8EBEEFF9 17B7BE43 60B08ED5 D6D6A3E8 A1D1937E 38D8C2C4 4FDFF252 D1BB67F1 A6BC5767 3FB506DD 48B2364B D80D2BDA AF0A1B4C 36034AF6 41047A60 DF60EFC3 A867DF55 316E8EEF 4669BE79 CB61B38C BC66831A 256FD2A0 5268E236 CC0C7795 BB0B4703 220216B9 5505262F C5BA3BBE B2BD0B28 2BB45A92 5CB36A04 C2D7FFA7 B5D0CF31 2CD99E8B 5BDEAE1D 9B64C2B0 EC63F226 756AA39C 026D930A 9C0906A9 EB0E363F 72076785 05005713 95BF4A82 E2B87A14 7BB12BAE 0CB61B38 92D28E9B E5D5BE0D 7CDCEFB7 0BDBDF21 86D3D2D4 F1D4E242 68DDB3F8 1FDA836E 81BE16CD F6B9265B 6FB077E1 18B74777 88085AE6 FF0F6A70 66063BCA 11010B5C 8F659EFF F862AE69 616BFFD3 166CCF45 A00AE278 D70DD2EE 4E048354 3903B3C2 A7672661 D06016F7 4969474D 3E6E77DB AED16A4A D9D65ADC 40DF0B66 37D83BF0 A9BCAE53 DEBB9EC5 47B2CF7F 30B5FFE9 BDBDF21C CABAC28A 53B39330 24B4A3A6 BAD03605 CDD70693 54DE5729 23D967BF B3667A2E C4614AB8 5D681B02 2A6F2B94 B40BBE37 C30C8EA1 5A05DF1B 2D02EF8D".split(' ');
		        
	        var table = new Array();
	        for (var i = 0; i < strTable.length; ++i) {
	          table[i] = parseInt("0x" + strTable[i]);
	        }
	
	        /* Number */
	        crc32 = function( /* String */ str, /* Number */ crc ) {
                if( crc == window.undefined ) crc = 0;
                var n = 0; //a number between 0 and 255
                var x = 0; //an hex number

                crc = crc ^ (-1);
                for( var i = 0, iTop = str.length; i < iTop; i++ ) {
                        n = ( crc ^ str.charCodeAt( i ) ) & 0xFF;
                        crc = ( crc >>> 8 ) ^ table[n];
                }
                return crc ^ (-1);
	        };
		})();
		
		function decimalToHexString(number) {
		    if (number < 0) {
		        number = 0xFFFFFFFF + number + 1;
		    }
		
		    return number.toString(16).toLowerCase();
		}
		
		
		
		
		
		
		function prepare_next_chunk(offset, offset_end, file){
			
			if(offset > file.size) return;
		
			if(offset_end > file.size) offset_end = file.size;
			
			var blob;
			
			if(file.slice)
				blob = file.slice(offset, offset_end);
			else if(file.webkitSlice)
		    	blob = file.webkitSlice(offset, offset_end);    	
		    else{
		    	alert("Your browser isn't supported!");
		     	throw "Browser not supported!";
		    }
		    
		    reader.readAsBinaryString(blob);  
			
		}
		
		// csv file's first row
		var header_row;
		
		function readBlob(opt_chunk) {

			disableUpload();
		    var files = document.getElementById('files').files;
		    document.getElementById('result').innerHTML = "";
		    if (!files.length) {
		      alert('Please select a file!');
		      return;
		    }
		
		    var file = files[0];
		    var chunk_size = parseInt(opt_chunk) || file.size;    
		 	var i = 0;
		    
			var pos_marker = 0;
		    //var parts = Math.floor(file.size / chunk_size)+1;    
		    
		    var selectedObj = document.getElementById("list_datasets");
		    if(selectedObj.length == 0) return;
		    var sel_dataset;
		    try{ sel_dataset = selectedObj.options[selectedObj.selectedIndex].value;}
		    catch(ex){}
		    if(sel_dataset === undefined || sel_dataset.length == 0) return;
		    
		    // If we use onloadend, we need to check the readyState.   
		     reader.onloadend = function(evt) {
		      if (evt.target.readyState == FileReader.DONE) { // DONE == 2
		        /*document.getElementById('byte_content').textContent = evt.target.result;*/
		       
		       if(i == 0) header_row = evt.target.result.split("\n")[0]; // first row of first chunk
		       
		        /*
				document.getElementById('file_info').textContent = 
									['Chunk size: ', chunk_size, ', Part num: ', ++i, ', total parts: ', parts,
									 ', total size: ', file.size, ' bytes'].join('');*/
				
		           
		    	//document.getElementById('result').textContent = parameters;
		    	var to_send = evt.target.result.slice(header_row.length, evt.target.result.lastIndexOf("\n"));
		    	
		    	var start = new Date().getTime();
		        var crc = crc32(to_send);
		        //var elapsed = new Date().getTime() - start;
		        //var mbits = (8.0 * evt.target.result.length * 1024.0 * 1024.0) / (elapsed * 1000);
		        
		      	//alert(mbits + " " + decimalToHexString(crc) + " dec: "+ crc);
		      	
		      	// encode special signs, unescape, base64-encode, compress one-liner
		        //cmprs = LZW.compress(window.btoa(unescape(encodeURIComponent(evt.target.result))));                   
				cmprs = LZW.compress(to_send);
				var sel_input = document.getElementById('list_inputs');				
				
		      	var parameters = 
		      	"chunk_id=" 		+ i++ + 
		      	"&chunk_size=" 		+ to_send.length + 		      	
		      	"&crc=" 			+ decimalToHexString(crc) + 
		      	//"&compressed_size=" + cmprs.length + 
		      	"&dataset=" 		+ sel_dataset + 
		      	"&input_queue="		+ sel_input.options[sel_input.selectedIndex].title +
		      	"&header="			+ header_row +
		      	"&data="			+ cmprs;
				//"&data="+ window.btoa(unescape(encodeURIComponent(evt.target.result)));
				
				var xhr = new XMLHttpRequest();
		      	xhr.open("POST", "upload");
		      	xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
		      	//var retry = 0;
		      	xhr.onreadystatechange = function() {
		    		//alert(data);		    		
		    		if(xhr.readyState == 4 && xhr.status == 200) {
		    			
	    				if(xhr.responseText.length > 0){ // if no response sent from server, 
	    											// either done, connection lost or server is down. 
	    											
	    					if(xhr.responseText.lastIndexOf("OK") == xhr.responseText.length - 2){
					    		pos_marker += to_send.length;
					    		prepare_next_chunk(pos_marker, pos_marker + chunk_size, 
					    			document.getElementById('files').files[0]);
					    		//document.getElementById('result').innerHTML = "";
					    		document.getElementById('result').innerHTML = 
					    			xhr.responseText.substring(0, xhr.responseText.lastIndexOf("OK"));
					    	}
					    	else{ // error or finish, end anyway and display message.
					    		document.getElementById('result').innerHTML = xhr.responseText;					    		
					    		return;
					    	}
			    		}	
			    		else{ // no response, assume retry after some delay.
			    			setTimeout(
			    				prepare_next_chunk(pos_marker, pos_marker + chunk_size, 
			    					document.getElementById('files').files[0])
			    				, 1000);
			    			document.getElementById('result').innerHTML = "Retrying to send " + i + "th chunk.";
			    		}		    		
		    		}
		    	};	
		      	
		      	xhr.setRequestHeader("Content-length", parameters.length);      	
		      	xhr.send(parameters);
		      
		      } // if FileReader == Done
		    };
		    
		    reader.onerror = function(evt) {
		    	alert("Error: " + evt.target.error.code);
		  	}
		
		    
		    prepare_next_chunk(pos_marker, chunk_size, file);
		 
		    
		  }
		  
		function create_input()		
		{												
			$.post(				 
			  "create_input",
	  		  $("#popform").serialize(),
			  function(data) {		
			  	if(this.responseText && this.responseText.indexOf("Failure") == 0){
			  		$("#output").html(data);
			  	}
			  	else{
				  	//$("#close-btn").click();
				  	location.reload(true);
			  	}	  	
							  	
			  }
			).error( function() {alert("Error while creating input table!");} );
		}
		  
		function create_dataset(){
			
			var a = $.ajax({
				type: "post",			 
			  	url: "create_dataset",
			  	data: $("#popform").serialize(),
			  	dataType: "text",
			  	cache: false,
			  	success:function(data){	
			  		if(data.indexOf("Failure") == 0){
				  		$("#output").html(data);
				  	}
				  	else{
				  		//$("#close-btn").click();
						location.reload(true);
				  	}
			  		
			     },
			     error:function(){			     	
			        $("#output").html("Error while creating dataset!");
			     },
			});			
		}
		  
		  
		  
		function recv_create_form_handler() {
			if(this.readyState == this.DONE && this.status == 200 && this.responseText.length > 0) {			    
		  		// success!
				$("#popup-wrapper").html(this.responseText);					
			}
		}
		  
		  
		
		function create_popup_form(type){		
				
			var request_url;
			
			if(type == "create_datasets"){
				//$("#popup-wrapper").css({"height":"300"});
				//$("#popup-wrapper").css({"width":"500"});
			 	request_url = "create_dataset?r=" + Math.random();
			}
			else if(type == "create_inputs") {
			
				request_url = "create_input?r=" + Math.random();
				//$("#popup-wrapper").css({"height":"500"});
				//$("#popup-wrapper").css({"width":"600"});
			}
			else return;
			
			var form = new XMLHttpRequest();
			form.onreadystatechange = recv_create_form_handler;
			form.open("GET", request_url);
			form.send();					
				
			return '<a href="#" id="close-btn">Close</a>' + '<p> Created by ' + type;
		}
		 
		
		
		function selection_change(tp) {
				
				var inputs_elem = document.getElementById("list_inputs");
				
				var selectObj = $("#list_datasets");				
				var idx_ds = selectObj.prop('selectedIndex');
				var which_ds = selectObj.prop('options')[idx_ds].text; // name of the dataset
					
				var inputs_data = complex[which_ds];
				
				if(tp == 1) // selected dataset has changed
				{
					
					if(idx_ds == selectObj.prop('options').length-1 && which_ds == "Create new..."){ // last item			
						$("#popup-wrapper").html(create_popup_form("create_datasets"));
						$("#open_popup").click();
						return;
					} 			
					
					// remove all subsequent items
					inputs_elem.innerHTML = "";
					
					for(var input in inputs_data){
						var option=document.createElement("option");
						
						option.text = input; // logical_name
						option.title = inputs_data[input][0]; // physical_table_name
						
						try{ // for IE earlier than version 8
						  inputs_elem.add(option,inputs_elem.options[null]);
						}
						catch (e) { inputs_elem.add(option,null); }			
					}						
					
					inputs_elem.innerHTML += "<OPTION>Create new...</OPTION>";
					inputs_elem.selectedIndex = 0;
					tp = 2;
				}
				
				
				if(tp == 2) // selected input table has changed
				{
					var idx_inp = inputs_elem.selectedIndex;
					var which_inp = inputs_elem.options[idx_inp].value;
					
					if(idx_inp == inputs_elem.options.length-1 && which_inp == "Create new..."){ // last item	
						$("#popup-wrapper").html(create_popup_form("create_inputs"));
						$("#open_popup").click();	
						return;
					}
					
					var input_columns = inputs_data[which_inp][1];
					
					var ifields = document.getElementById('input_fields');
					ifields.value = "";
					var tablerow = document.getElementById("text_input_row");
					
					tablerow.innerHTML = "";
					
					var colname = document.getElementById("columns_name");
					colname.innerHTML = "";
					
					if(input_columns === undefined) 
						return;
					
					for(var i=0; i < input_columns.length; i++){				
						tablerow.innerHTML += "<td><textarea name=\"" + input_columns[i] + "\"></textarea></td>";	
						colname.innerHTML += "<td><center>" + input_columns[i] + "</center></td>";	
						ifields.value += input_columns[i] + ',';
					}
					ifields.value = ifields.value.substr(0, ifields.value.length - 1);
				}
				
			}


		function initPage(){
									
			var select_ds = document.getElementById("list_datasets");
			var select_inputs = document.getElementById("list_inputs");
			
			select_ds.innerHTML = "";
			select_inputs.innerHTML = "";
			
						
			if(Object.size(complex) > 0){
				enableUpload();
				var dataset;
				
				for(dataset in complex){
					select_ds.innerHTML += "<OPTION>" + dataset + "</OPTION>";						
				}
				
				select_ds.innerHTML += "<OPTION>Create new...</OPTION>";
				select_ds.selectedIndex = 0;
				
				if(typeof selection_change == "function")
					selection_change(1);				
			}
				
			else {
				disableUpload();
				select_ds.innerHTML += "<OPTION>Create new...</OPTION>";
				selected_dataset_change(select_ds);
				//var sel_ds = select_ds.options[select_ds.selectedIndex].value;
				//for(i_name in input_tables[sel_ds]){
				//	select_inputs.innerHTML += "<OPTION>" + i_name + "</OPTION>";
				//}				
			}		
						
		}
		
		function doSubmit(){
			var sel_inputs = document.getElementById('list_inputs');
			
			var post_data = $("#main_form").serialize() + '&input_queue=' +			
				sel_inputs.options[sel_inputs.selectedIndex].title;				
			
			$("#clear_form").click();			
			
			$.post(				 
			  $("#main_form").prop('action'),
	  		  post_data,
			  function(data) {		
			  	if(data.length > 0){
			  		$("#popup-wrapper").html('<p>' + data 
			  		+ '</p><p><a href="#" id="close-btn">Close</a></p>');
			  	}
			  	else{
				  	$("#popup-wrapper").html('Done.<p><a href="#" id="close-btn">Close</a></p>');
			  	}	  	
			  	$("#open_popup").click();
			  }
			).error( function() {alert("Error while creating input table!");} );
			
		}
		
		function enableUpload(){
			document.getElementById("file_upload").disabled=false;
		}
		function disableUpload(){
			document.getElementById("file_upload").disabled=true;
		}
		
		$(function () {
		    $('#popup-wrapper').modalPopLite({ openButton: '#open_popup', closeButton: '#close-btn', isModal: true });
		});
