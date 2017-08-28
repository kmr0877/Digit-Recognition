# Digit-Recognition

You   will   see   two   files:   train.py      and      hw1.py Now   run      train.py      by   typing
python3   train.py
When   run   for   the   first   time,      train.py      should   create   a   new   folder   called      data      and   download   a copy   of   the   MNIST   dataset   into   this   folder.   All   subsequent   runs   of      train.py      will   use   this   local data.   (Don't   worry   about   the      ValueError      at   this   stage.)

The   file      train.py      contains   the   TensorFlow   code   required   to   create   a   session,   build   the   graph, and   run   training   and   test   iterations.   It   has   been   provided   to   assist   you   with   the   testing   and evaluation   of   your   model.   While   it   is   not   required   for   this   assignment   to   have   a   detailed understanding   of   this   code,   it   will   be   useful   when   implementing   your   own   models,   and   for   later assignments.

The   file      train.py      calls   functions   defined   in   hw1.py      and   should   not   be   modified   during   the course   of   the   assignment.   A   submission   that   does   not   run   correctly   when      train.py      is   called   will lose   marks.   The   only   situation   where   you   should   modify      train.py      is   when   you   need   to   switch between   different   network   architectures.  

This   can   be   done   by   setting   the   global   variable   on   line 7:
network   =   "none"
to   any   of   the   following   values: network   =   "onelayer"
network   =   "twolayer"
network   =   "conv"


The   file      hw1.py      contains   function   definitions   for   the   three   networks   to   be   created.   You   may   also define   helper   functions   in   this   file   if   necessary,   as   long   as   the   original   function   names   and arguments   are   not   modified.   Changing   the   function   name,   argument   list,   or   return   value   will cause   all   tests   to   fail   for   that   function.   Your   marks   will   be   automatically   generated   by   a   test script,   which   will   evaluate   the   correctness   of   the   implemented   networks.   For   this   reason,   it   is important   that   you   stick   to   the   specification   exactly.   Networks   that   do   not   meet   the   specifications but   otherwise   function   accurately,   will   be   marked   as   incorrect.
