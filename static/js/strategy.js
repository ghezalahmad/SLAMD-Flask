$("#strategy").change(function(){
  var abc = $('#strategy').val();
  if (abc == '1'){
    //alert('hello');
    //$document.getElementByID('first').show();
      $('#sigma').hide();
      $('#prediction').hide();
  } else if (abc == '2'){
      $('#prediction').hide();
      $('#sigma').hide();
  } else if (abc == '3'){
      $('#prediction').hide();
      $('#sigma').show();
  } else if (abc == '4'){
      $('#sigma').hide();
      $('#prediction').show();
  } else if (abc == '5'){
      $('#sigma').show();
      $('#prediction').show();
  } else {
      alert('select a option');
  }

});
