$(document).ready(function() {
    $('a.abstract').click(function() {
        $(this).parent().parent().find(".abstract.hidden").toggleClass('open');
    });
    $('a.bibtex').click(function() {
        $(this).parent().parent().find(".bibtex.hidden").toggleClass('open');
    });
    $('.navbar-nav').find('a').removeClass('waves-effect waves-light');
});

console.log(document.querySelector('#main-canvas'));

$('#startbtn').on('click', (e) => {
    console.log('START!');
    console.log(`We also have this object to draw on ${$('#main-canvas')}`);
    console.log(`Here's the tensorflow thingy! ${tf}`);
})