*===========================================================
* MAIN SCRIPT
*
*---------------------------------------------------
'reinit'
*---------------------------------------------------
'open hs_mask.pctl'
'set vpage 0.0 11.0 0.00 8.50'
'set parea 0.5 10.5 1.50 7.50'
'set lon 45 405'
'set lat -90 90'
'set xlint 30'
'set ylint 10'
'set grid on'
'set mproj scaled'
*'set mpdset lowres'
'set poli off'
'set map 1 1 3'
'set xlab on'
'set ylab on'
'set xlopts 1 3 0.12'
'set ylopts 1 3 0.12'
'set gxout grfill'
'set rbcols 1   2   3   4   5   6   7   8   9  10  11    1'
'set clevs   0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5'
'set grads off'
'd basin'
 label(190.0,95.0,"OMIP Basin mask (Griffies et al. (2016) section G6)",50,0,0.15)
'close 1'
'run cbarn2.gs 0.9 0 5.5 1.0 0 1'
*------------------------------------------------------------
'printim basin_mask_omip.png png white'
*------------------------------------------------------------
*'enable print 
*'disable print'
*'!gxeps -c -i basin_mask.gm -o basin_mask.eps'
*'!rm basin_mask.gm'
*------------------------------------------------------------
function label(x,y,lab,len,angle,size,justify)
if(size='' | size='size');size=0.10;endif;
if(justify='' | justify='justify');justify='c';endif;
size2=size*1.2
'set strsiz ' size ' ' size2 
'set string 1 ' justify ' 3 ' angle
w = size*len/2
h = (size2*1.4)/2 
'query w2xy ' x ' ' y
 x = subwrd(result,3)
 y = subwrd(result,6)
'set line 0'
if(angle=0)
 'draw recf '%(x-w)%' '%(y-h*1.2)%' '%(x+w)%' '%(y+h)
endif
'draw string ' x ' ' y ' ' lab
'set line 1'
'set string 1 c 3 0' 
*------------------------------------------
function shade2(type)
 'set rgb 40   0   0   0'
 'set rgb 41  40  40  40'
 'set rgb 42  60  60  60'
 'set rgb 43  80  80  80'
 'set rgb 44 100 100 100'
 'set rgb 45 120 120 120'
 'set rgb 46 140 140 140'
 'set rgb 47 160 160 160'
if(type='topo')
 'set rbcols 41 42 43 44 45 46 47'
 'set clevs    1000 2000 3000 4000 5000 6000'
endif
if(type='mix')
 'set rbcols 41 42 43 44 45 46 47'
 'set clevs    50 100 150 200 250 300'
endif
