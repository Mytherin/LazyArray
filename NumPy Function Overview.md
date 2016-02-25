

#Pipelineable (element-wise)
--Trigonometric Functions--
sin
cos
tan
arcsin
arccos
arctan
hypot
arctan2
deg2rad
radians
unwrap
rad2deg

--Hyperbolic Functions--
sinh
cosh
tanh
arcsinh
arccosh
arctanh

--Rounding--
around
round
rint
fix
floor
ceil
trunc

--Exponents and logarithms--
exp
expm1
exp2
log
log10
log2
log1p
logaddexp1
logaddexp2

--Other special functions--
sinc

--Floating point operations--
signbit
copysign
frexp
ldexp

--Arithmetic operations--
add
reciprocal
negative
multiply
divide
power
subtract
true_divide
floor_divide
fmod
mod
modf
remainder

--Handling complex numbers--
angle
real
imag

--Miscellaneous--
clip
sqrt
square
absolute
fabs
sign
nan_to_num

--Binary Operations--
bitwise_and
bitwise_or
bitwise_xor
invert
left_shift
right_shift

--String Operations--
add
multiply
mod
capitalize
center
decode
encode
ljust
lower
lstrip
partition
replace
rjust
rpartition
rstrip
split
splitlines
strip
swapcase
title
translate
upper
zfill
equal
not_equal
greater_equal
less_equal
greater
less
find
index
isalpha
isdecimal
isdigit
islower
isnumeric
isspace
istitle
isupper
rfind
rindex
startswith

--Indexing--
nonzero
where
choose

--Logic Functions--
isfinite
isinf
isnan
isneginf
isposinf
logical_and
logical_or
logical_not
logical_xor
isclose
greater
greater_equal
less
less_equal
equal
not_equal


--Array manipulation routines--
copyto


# Aggregations (cumulatively pipelinable)
prod
sum
nansum
maximum
minimum
fmax
fmin

numpy.core.defchararray.join

Join all strings on a character, this is pipelinable because we can join each of the pipelines and then join the different pipelines together.

unique

First compute unique on pipeline chunks, then compute unique on the resulting set of unique values. So each pipe generates a set of unique values in the pipe, that is at most as big as the pipe itself. Then we perform unique on the entirety of those values. 

--Logic Functions--
allclose
array_equal
array_equiv

--Sorting, search and counting--
argmax
nanargmax
argmin
nanargmin
argwhere
nonzero
flatnonzero

# Almost pipelinable
diff
ediff1d

For every value, compute the difference to the previous value (i.e. compute deltas). [1,2,3,4,5] => [1,1,1,1,1]. We can compute everything pipelined except for the first value, which relies on the value of the previous pipeline. [1,2,3] [4,5,6] => [1,1,1] [?,1,1], we need to know 3.

This is a problem because to get 3, we need to compute the previous box, for which we need to compute the previous box. However, if we are only interested in, say, the 47th box, we only need to compute box 46/47, not box 1-47.

cross
Computes cross product in three-dimensional space (so vectors must be 3 elements max), but can be computed over multiple vectors -> we can pipeline that because it's independent

real_if_close
This basically checks 'if all the complex parts are close to zero, return a float array instead', so we need to first check all elements before we can do the conversion, but the checking and converting itself is pipelinable, just with a 'stop pipelining and wait for all the checks to finish before continuing' in the middle.

shuffle
permutation

MAYBE this introduces bias, I am not sure. For each element in a pipeline, you randomly select a location to place the element. If there is already an element at that location, you do random() again until you find an empty location. 

sort, lexsort, argsort, msort, sort_complex


# Not pipelinable
cumprod
cumsum

Every value relies on the previous value. Cumsum is [1, 2, 3, 4] => [1, 3, 6, 10]. We might be able to do semi-pipelines. Suppose two blocks: [1,2] [3,4], we can first compute [1, 3] [3, 7]. Then we add +3 to every element in the second block.


# Not sure
gradient -> don't think this is pipelinable, it has something to do with the derivative
trapz -> Trapezoidal rule, no clue, I think at least partially pipelinable though
i0 -> Bessel function, wh atever that may be
conj -> complex conjugate, probably not
convolve -> probably not
interp -> Probably not



TODO: 
Linear Algebra: http://docs.scipy.org/doc/numpy/reference/routines.linalg.html
Statistics: http://docs.scipy.org/doc/numpy/reference/routines.statistics.html