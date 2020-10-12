#! /usr/bin/perl -pi.orig

s/\bConditionallyOptimize\b/MayOptimize/g;
s/\@may_assume_inbounds\b/\@maybe_inbounds/g;
s/\@may_vectorize\b/\@maybe_vectorized/g;
