/*
* Copyright (c) 2013 Battelle Memorial Institute.
* All rights reserved.
*
* The software in this package is published under the terms of the BSD
* style license a copy of which has been included with this distribution in
* the LICENSE.txt file.
*
* Created on 2013 by John Feo
*/

#include <sys/time.h>

double timer(void)
{

//     return the time since the last call to timer.

//     must be initialized by calling once and throwing away the
//     value 
//     ... use cpu time on multi-user machines
//     ... use elapsed time on dedicated or single user machines.
//
//mdc*if unix
//      real*4 dtime, tt(2)
//      timer = dble(dtime(tt))
//mdc*elseif tcgmsg
//Id: timer.F,v 1.1 2005-03-08 23:58:03 d3g293 Exp $

  const double million = 1.0e6;

  static struct timeval tvlast;
  static int initted = 0;

  struct timeval tv;
  struct timezone tz;

  double ret, t0, t1;

  gettimeofday(&tv, &tz);

  if (!initted) {
    tvlast = tv;
    initted = 1;
  }

  t0 = tvlast.tv_sec * million + tvlast.tv_usec;
  t1 = tv.tv_sec * million + tv.tv_usec;

  ret = (t1 - t0) / million;
  tvlast = tv;

  return ret;
}
