Ш╟
├Ч
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
╝
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.9.22v2.9.1-132-g18960c44ad38ЙН
|
Adam/exit_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/exit_4/bias/v
u
&Adam/exit_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/exit_4/bias/v*
_output_shapes
:
*
dtype0
Д
Adam/exit_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T
*%
shared_nameAdam/exit_4/kernel/v
}
(Adam/exit_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/exit_4/kernel/v*
_output_shapes

:T
*
dtype0
|
Adam/exit_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/exit_3/bias/v
u
&Adam/exit_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/exit_3/bias/v*
_output_shapes
:
*
dtype0
Д
Adam/exit_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x
*%
shared_nameAdam/exit_3/kernel/v
}
(Adam/exit_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/exit_3/kernel/v*
_output_shapes

:x
*
dtype0
|
Adam/exit_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/exit_2/bias/v
u
&Adam/exit_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/exit_2/bias/v*
_output_shapes
:
*
dtype0
Е
Adam/exit_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*%
shared_nameAdam/exit_2/kernel/v
~
(Adam/exit_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/exit_2/kernel/v*
_output_shapes
:	А
*
dtype0
|
Adam/exit_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/exit_1/bias/v
u
&Adam/exit_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/exit_1/bias/v*
_output_shapes
:
*
dtype0
Е
Adam/exit_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*%
shared_nameAdam/exit_1/kernel/v
~
(Adam/exit_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/exit_1/kernel/v*
_output_shapes
:	А
*
dtype0
t
Adam/F6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*
shared_nameAdam/F6/bias/v
m
"Adam/F6/bias/v/Read/ReadVariableOpReadVariableOpAdam/F6/bias/v*
_output_shapes
:T*
dtype0
|
Adam/F6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*!
shared_nameAdam/F6/kernel/v
u
$Adam/F6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/F6/kernel/v*
_output_shapes

:xT*
dtype0
t
Adam/F5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_nameAdam/F5/bias/v
m
"Adam/F5/bias/v/Read/ReadVariableOpReadVariableOpAdam/F5/bias/v*
_output_shapes
:x*
dtype0
}
Adam/F5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Аx*!
shared_nameAdam/F5/kernel/v
v
$Adam/F5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/F5/kernel/v*
_output_shapes
:	Аx*
dtype0
t
Adam/C3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/C3/bias/v
m
"Adam/C3/bias/v/Read/ReadVariableOpReadVariableOpAdam/C3/bias/v*
_output_shapes
:*
dtype0
Д
Adam/C3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/C3/kernel/v
}
$Adam/C3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/C3/kernel/v*&
_output_shapes
:*
dtype0
t
Adam/C1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/C1/bias/v
m
"Adam/C1/bias/v/Read/ReadVariableOpReadVariableOpAdam/C1/bias/v*
_output_shapes
:*
dtype0
Д
Adam/C1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/C1/kernel/v
}
$Adam/C1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/C1/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/exit_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/exit_4/bias/m
u
&Adam/exit_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/exit_4/bias/m*
_output_shapes
:
*
dtype0
Д
Adam/exit_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T
*%
shared_nameAdam/exit_4/kernel/m
}
(Adam/exit_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/exit_4/kernel/m*
_output_shapes

:T
*
dtype0
|
Adam/exit_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/exit_3/bias/m
u
&Adam/exit_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/exit_3/bias/m*
_output_shapes
:
*
dtype0
Д
Adam/exit_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x
*%
shared_nameAdam/exit_3/kernel/m
}
(Adam/exit_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/exit_3/kernel/m*
_output_shapes

:x
*
dtype0
|
Adam/exit_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/exit_2/bias/m
u
&Adam/exit_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/exit_2/bias/m*
_output_shapes
:
*
dtype0
Е
Adam/exit_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*%
shared_nameAdam/exit_2/kernel/m
~
(Adam/exit_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/exit_2/kernel/m*
_output_shapes
:	А
*
dtype0
|
Adam/exit_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/exit_1/bias/m
u
&Adam/exit_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/exit_1/bias/m*
_output_shapes
:
*
dtype0
Е
Adam/exit_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*%
shared_nameAdam/exit_1/kernel/m
~
(Adam/exit_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/exit_1/kernel/m*
_output_shapes
:	А
*
dtype0
t
Adam/F6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*
shared_nameAdam/F6/bias/m
m
"Adam/F6/bias/m/Read/ReadVariableOpReadVariableOpAdam/F6/bias/m*
_output_shapes
:T*
dtype0
|
Adam/F6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*!
shared_nameAdam/F6/kernel/m
u
$Adam/F6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/F6/kernel/m*
_output_shapes

:xT*
dtype0
t
Adam/F5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_nameAdam/F5/bias/m
m
"Adam/F5/bias/m/Read/ReadVariableOpReadVariableOpAdam/F5/bias/m*
_output_shapes
:x*
dtype0
}
Adam/F5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Аx*!
shared_nameAdam/F5/kernel/m
v
$Adam/F5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/F5/kernel/m*
_output_shapes
:	Аx*
dtype0
t
Adam/C3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/C3/bias/m
m
"Adam/C3/bias/m/Read/ReadVariableOpReadVariableOpAdam/C3/bias/m*
_output_shapes
:*
dtype0
Д
Adam/C3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/C3/kernel/m
}
$Adam/C3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/C3/kernel/m*&
_output_shapes
:*
dtype0
t
Adam/C1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/C1/bias/m
m
"Adam/C1/bias/m/Read/ReadVariableOpReadVariableOpAdam/C1/bias/m*
_output_shapes
:*
dtype0
Д
Adam/C1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/C1/kernel/m
}
$Adam/C1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/C1/kernel/m*&
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_7
[
total_7/Read/ReadVariableOpReadVariableOptotal_7*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
n
exit_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameexit_4/bias
g
exit_4/bias/Read/ReadVariableOpReadVariableOpexit_4/bias*
_output_shapes
:
*
dtype0
v
exit_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T
*
shared_nameexit_4/kernel
o
!exit_4/kernel/Read/ReadVariableOpReadVariableOpexit_4/kernel*
_output_shapes

:T
*
dtype0
n
exit_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameexit_3/bias
g
exit_3/bias/Read/ReadVariableOpReadVariableOpexit_3/bias*
_output_shapes
:
*
dtype0
v
exit_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x
*
shared_nameexit_3/kernel
o
!exit_3/kernel/Read/ReadVariableOpReadVariableOpexit_3/kernel*
_output_shapes

:x
*
dtype0
n
exit_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameexit_2/bias
g
exit_2/bias/Read/ReadVariableOpReadVariableOpexit_2/bias*
_output_shapes
:
*
dtype0
w
exit_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*
shared_nameexit_2/kernel
p
!exit_2/kernel/Read/ReadVariableOpReadVariableOpexit_2/kernel*
_output_shapes
:	А
*
dtype0
n
exit_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameexit_1/bias
g
exit_1/bias/Read/ReadVariableOpReadVariableOpexit_1/bias*
_output_shapes
:
*
dtype0
w
exit_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*
shared_nameexit_1/kernel
p
!exit_1/kernel/Read/ReadVariableOpReadVariableOpexit_1/kernel*
_output_shapes
:	А
*
dtype0
f
F6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*
shared_name	F6/bias
_
F6/bias/Read/ReadVariableOpReadVariableOpF6/bias*
_output_shapes
:T*
dtype0
n
	F6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*
shared_name	F6/kernel
g
F6/kernel/Read/ReadVariableOpReadVariableOp	F6/kernel*
_output_shapes

:xT*
dtype0
f
F5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_name	F5/bias
_
F5/bias/Read/ReadVariableOpReadVariableOpF5/bias*
_output_shapes
:x*
dtype0
o
	F5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Аx*
shared_name	F5/kernel
h
F5/kernel/Read/ReadVariableOpReadVariableOp	F5/kernel*
_output_shapes
:	Аx*
dtype0
f
C3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	C3/bias
_
C3/bias/Read/ReadVariableOpReadVariableOpC3/bias*
_output_shapes
:*
dtype0
v
	C3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	C3/kernel
o
C3/kernel/Read/ReadVariableOpReadVariableOp	C3/kernel*&
_output_shapes
:*
dtype0
f
C1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	C1/bias
_
C1/bias/Read/ReadVariableOpReadVariableOpC1/bias*
_output_shapes
:*
dtype0
v
	C1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	C1/kernel
o
C1/kernel/Read/ReadVariableOpReadVariableOp	C1/kernel*&
_output_shapes
:*
dtype0

NoOpNoOp
ЁБ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*кБ
valueЯБBЫБ BУБ
║
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
О
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses* 
╚
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op*
О
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
О
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 
ж
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias*
ж
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias*
ж
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

Pkernel
Qbias*
ж
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias*
ж
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias*
ж
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias*
z
0
1
+2
,3
@4
A5
H6
I7
P8
Q9
X10
Y11
`12
a13
h14
i15*
z
0
1
+2
,3
@4
A5
H6
I7
P8
Q9
X10
Y11
`12
a13
h14
i15*
* 
░
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
otrace_0
ptrace_1
qtrace_2
rtrace_3* 
6
strace_0
ttrace_1
utrace_2
vtrace_3* 
* 
Д
witer

xbeta_1

ybeta_2
	zdecay
{learning_ratem mА+mБ,mВ@mГAmДHmЕImЖPmЗQmИXmЙYmК`mЛamМhmНimОvПvР+vС,vТ@vУAvФHvХIvЦPvЧQvШXvЩYvЪ`vЫavЬhvЭivЮ*

|serving_default* 

0
1*

0
1*
* 
Х
}non_trainable_variables

~layers
metrics
 Аlayer_regularization_losses
Бlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Вtrace_0* 

Гtrace_0* 
YS
VARIABLE_VALUE	C1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEC1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 

Йtrace_0* 

Кtrace_0* 

+0
,1*

+0
,1*
* 
Ш
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Рtrace_0* 

Сtrace_0* 
YS
VARIABLE_VALUE	C3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEC3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

Чtrace_0* 

Шtrace_0* 
* 
* 
* 
Ц
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 
,
Юtrace_0
Яtrace_1
аtrace_2* 
,
бtrace_0
вtrace_1
гtrace_2* 

@0
A1*

@0
A1*
* 
Ш
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

йtrace_0* 

кtrace_0* 
YS
VARIABLE_VALUE	F5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEF5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*

H0
I1*
* 
Ш
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

░trace_0* 

▒trace_0* 
YS
VARIABLE_VALUE	F6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEF6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

P0
Q1*

P0
Q1*
* 
Ш
▓non_trainable_variables
│layers
┤metrics
 ╡layer_regularization_losses
╢layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

╖trace_0* 

╕trace_0* 
]W
VARIABLE_VALUEexit_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEexit_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

X0
Y1*

X0
Y1*
* 
Ш
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

╛trace_0* 

┐trace_0* 
]W
VARIABLE_VALUEexit_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEexit_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

`0
a1*

`0
a1*
* 
Ш
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

┼trace_0* 

╞trace_0* 
]W
VARIABLE_VALUEexit_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEexit_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

h0
i1*

h0
i1*
* 
Ш
╟non_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

╠trace_0* 

═trace_0* 
]W
VARIABLE_VALUEexit_4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEexit_4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
Z
0
1
2
3
4
5
6
7
	8

9
10
11*
L
╬0
╧1
╨2
╤3
╥4
╙5
╘6
╒7
╓8*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
╫	variables
╪	keras_api

┘total

┌count*
<
█	variables
▄	keras_api

▌total

▐count*
<
▀	variables
р	keras_api

сtotal

тcount*
<
у	variables
ф	keras_api

хtotal

цcount*
<
ч	variables
ш	keras_api

щtotal

ъcount*
M
ы	variables
ь	keras_api

эtotal

юcount
я
_fn_kwargs*
M
Ё	variables
ё	keras_api

Єtotal

єcount
Ї
_fn_kwargs*
M
ї	variables
Ў	keras_api

ўtotal

°count
∙
_fn_kwargs*
M
·	variables
√	keras_api

№total

¤count
■
_fn_kwargs*

┘0
┌1*

╫	variables*
UO
VARIABLE_VALUEtotal_84keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_84keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

▌0
▐1*

█	variables*
UO
VARIABLE_VALUEtotal_74keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_74keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

с0
т1*

▀	variables*
UO
VARIABLE_VALUEtotal_64keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_64keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

х0
ц1*

у	variables*
UO
VARIABLE_VALUEtotal_54keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_54keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*

щ0
ъ1*

ч	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*

э0
ю1*

ы	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Є0
є1*

Ё	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

ў0
°1*

ї	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

№0
¤1*

·	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
|v
VARIABLE_VALUEAdam/C1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/C1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/C3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/C3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/F5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/F5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/F6/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/F6/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/exit_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/exit_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/exit_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/exit_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/exit_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/exit_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/exit_4/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/exit_4/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/C1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/C1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/C3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/C3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/F5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/F5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/F6/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/F6/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/exit_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/exit_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/exit_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/exit_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/exit_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/exit_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/exit_4/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/exit_4/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
И
serving_default_inputPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
╬
StatefulPartitionedCallStatefulPartitionedCallserving_default_input	C1/kernelC1/bias	C3/kernelC3/bias	F5/kernelF5/bias	F6/kernelF6/biasexit_4/kernelexit_4/biasexit_3/kernelexit_3/biasexit_2/kernelexit_2/biasexit_1/kernelexit_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:         
:         
:         
:         
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_80421
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
є
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameC1/kernel/Read/ReadVariableOpC1/bias/Read/ReadVariableOpC3/kernel/Read/ReadVariableOpC3/bias/Read/ReadVariableOpF5/kernel/Read/ReadVariableOpF5/bias/Read/ReadVariableOpF6/kernel/Read/ReadVariableOpF6/bias/Read/ReadVariableOp!exit_1/kernel/Read/ReadVariableOpexit_1/bias/Read/ReadVariableOp!exit_2/kernel/Read/ReadVariableOpexit_2/bias/Read/ReadVariableOp!exit_3/kernel/Read/ReadVariableOpexit_3/bias/Read/ReadVariableOp!exit_4/kernel/Read/ReadVariableOpexit_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_8/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_7/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$Adam/C1/kernel/m/Read/ReadVariableOp"Adam/C1/bias/m/Read/ReadVariableOp$Adam/C3/kernel/m/Read/ReadVariableOp"Adam/C3/bias/m/Read/ReadVariableOp$Adam/F5/kernel/m/Read/ReadVariableOp"Adam/F5/bias/m/Read/ReadVariableOp$Adam/F6/kernel/m/Read/ReadVariableOp"Adam/F6/bias/m/Read/ReadVariableOp(Adam/exit_1/kernel/m/Read/ReadVariableOp&Adam/exit_1/bias/m/Read/ReadVariableOp(Adam/exit_2/kernel/m/Read/ReadVariableOp&Adam/exit_2/bias/m/Read/ReadVariableOp(Adam/exit_3/kernel/m/Read/ReadVariableOp&Adam/exit_3/bias/m/Read/ReadVariableOp(Adam/exit_4/kernel/m/Read/ReadVariableOp&Adam/exit_4/bias/m/Read/ReadVariableOp$Adam/C1/kernel/v/Read/ReadVariableOp"Adam/C1/bias/v/Read/ReadVariableOp$Adam/C3/kernel/v/Read/ReadVariableOp"Adam/C3/bias/v/Read/ReadVariableOp$Adam/F5/kernel/v/Read/ReadVariableOp"Adam/F5/bias/v/Read/ReadVariableOp$Adam/F6/kernel/v/Read/ReadVariableOp"Adam/F6/bias/v/Read/ReadVariableOp(Adam/exit_1/kernel/v/Read/ReadVariableOp&Adam/exit_1/bias/v/Read/ReadVariableOp(Adam/exit_2/kernel/v/Read/ReadVariableOp&Adam/exit_2/bias/v/Read/ReadVariableOp(Adam/exit_3/kernel/v/Read/ReadVariableOp&Adam/exit_3/bias/v/Read/ReadVariableOp(Adam/exit_4/kernel/v/Read/ReadVariableOp&Adam/exit_4/bias/v/Read/ReadVariableOpConst*T
TinM
K2I	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__traced_save_81101
т

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	C1/kernelC1/bias	C3/kernelC3/bias	F5/kernelF5/bias	F6/kernelF6/biasexit_1/kernelexit_1/biasexit_2/kernelexit_2/biasexit_3/kernelexit_3/biasexit_4/kernelexit_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_8count_8total_7count_7total_6count_6total_5count_5total_4count_4total_3count_3total_2count_2total_1count_1totalcountAdam/C1/kernel/mAdam/C1/bias/mAdam/C3/kernel/mAdam/C3/bias/mAdam/F5/kernel/mAdam/F5/bias/mAdam/F6/kernel/mAdam/F6/bias/mAdam/exit_1/kernel/mAdam/exit_1/bias/mAdam/exit_2/kernel/mAdam/exit_2/bias/mAdam/exit_3/kernel/mAdam/exit_3/bias/mAdam/exit_4/kernel/mAdam/exit_4/bias/mAdam/C1/kernel/vAdam/C1/bias/vAdam/C3/kernel/vAdam/C3/bias/vAdam/F5/kernel/vAdam/F5/bias/vAdam/F6/kernel/vAdam/F6/bias/vAdam/exit_1/kernel/vAdam/exit_1/bias/vAdam/exit_2/kernel/vAdam/exit_2/bias/vAdam/exit_3/kernel/vAdam/exit_3/bias/vAdam/exit_4/kernel/vAdam/exit_4/bias/v*S
TinL
J2H*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_81324Яч	
┬
Ф
&__inference_exit_1_layer_call_fn_80791

inputs
unknown:	А

	unknown_0:

identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_1_layer_call_and_return_conditional_losses_79938o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б

є
A__inference_exit_2_layer_call_and_return_conditional_losses_80822

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
х8
┘
@__inference_model_layer_call_and_return_conditional_losses_80370	
input"
c1_80321:
c1_80323:"
c3_80327:
c3_80329:
f5_80336:	Аx
f5_80338:x
f6_80341:xT
f6_80343:T
exit_4_80346:T

exit_4_80348:

exit_3_80351:x

exit_3_80353:

exit_2_80356:	А

exit_2_80358:

exit_1_80361:	А

exit_1_80363:

identity

identity_1

identity_2

identity_3ИвC1/StatefulPartitionedCallвC3/StatefulPartitionedCallвF5/StatefulPartitionedCallвF6/StatefulPartitionedCallвexit_1/StatefulPartitionedCallвexit_2/StatefulPartitionedCallвexit_3/StatefulPartitionedCallвexit_4/StatefulPartitionedCall▀
C1/StatefulPartitionedCallStatefulPartitionedCallinputc1_80321c1_80323*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_C1_layer_call_and_return_conditional_losses_79795╙
S2/PartitionedCallPartitionedCall#C1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_S2_layer_call_and_return_conditional_losses_79762ї
C3/StatefulPartitionedCallStatefulPartitionedCallS2/PartitionedCall:output:0c3_80327c3_80329*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_C3_layer_call_and_return_conditional_losses_79813╙
S4/PartitionedCallPartitionedCall#C3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_S4_layer_call_and_return_conditional_losses_79774╬
Flatten/PartitionedCallPartitionedCallS4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79826╪
Flatten/PartitionedCall_1PartitionedCall#C3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79833╪
Flatten/PartitionedCall_2PartitionedCall#C1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79840Є
F5/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0f5_80336f5_80338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_F5_layer_call_and_return_conditional_losses_79853ї
F6/StatefulPartitionedCallStatefulPartitionedCall#F5/StatefulPartitionedCall:output:0f6_80341f6_80343*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_F6_layer_call_and_return_conditional_losses_79870Е
exit_4/StatefulPartitionedCallStatefulPartitionedCall#F6/StatefulPartitionedCall:output:0exit_4_80346exit_4_80348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_4_layer_call_and_return_conditional_losses_79887Е
exit_3/StatefulPartitionedCallStatefulPartitionedCall#F5/StatefulPartitionedCall:output:0exit_3_80351exit_3_80353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_3_layer_call_and_return_conditional_losses_79904Д
exit_2/StatefulPartitionedCallStatefulPartitionedCall"Flatten/PartitionedCall_1:output:0exit_2_80356exit_2_80358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_2_layer_call_and_return_conditional_losses_79921Д
exit_1/StatefulPartitionedCallStatefulPartitionedCall"Flatten/PartitionedCall_2:output:0exit_1_80361exit_1_80363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_1_layer_call_and_return_conditional_losses_79938v
IdentityIdentity'exit_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_1Identity'exit_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_2Identity'exit_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_3Identity'exit_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
╛
NoOpNoOp^C1/StatefulPartitionedCall^C3/StatefulPartitionedCall^F5/StatefulPartitionedCall^F6/StatefulPartitionedCall^exit_1/StatefulPartitionedCall^exit_2/StatefulPartitionedCall^exit_3/StatefulPartitionedCall^exit_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 28
C1/StatefulPartitionedCallC1/StatefulPartitionedCall28
C3/StatefulPartitionedCallC3/StatefulPartitionedCall28
F5/StatefulPartitionedCallF5/StatefulPartitionedCall28
F6/StatefulPartitionedCallF6/StatefulPartitionedCall2@
exit_1/StatefulPartitionedCallexit_1/StatefulPartitionedCall2@
exit_2/StatefulPartitionedCallexit_2/StatefulPartitionedCall2@
exit_3/StatefulPartitionedCallexit_3/StatefulPartitionedCall2@
exit_4/StatefulPartitionedCallexit_4/StatefulPartitionedCall:V R
/
_output_shapes
:         

_user_specified_nameinput
─
^
B__inference_Flatten_layer_call_and_return_conditional_losses_79826

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
О
Y
=__inference_S2_layer_call_and_return_conditional_losses_80679

inputs
identityл
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ф

ю
=__inference_F6_layer_call_and_return_conditional_losses_80782

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Tr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         TP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         Ta
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         Tw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
¤

Ў
=__inference_C1_layer_call_and_return_conditional_losses_79795

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Э

Є
A__inference_exit_3_layer_call_and_return_conditional_losses_80842

inputs0
matmul_readvariableop_resource:x
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
О
Y
=__inference_S4_layer_call_and_return_conditional_losses_79774

inputs
identityл
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
х8
┘
@__inference_model_layer_call_and_return_conditional_losses_80318	
input"
c1_80269:
c1_80271:"
c3_80275:
c3_80277:
f5_80284:	Аx
f5_80286:x
f6_80289:xT
f6_80291:T
exit_4_80294:T

exit_4_80296:

exit_3_80299:x

exit_3_80301:

exit_2_80304:	А

exit_2_80306:

exit_1_80309:	А

exit_1_80311:

identity

identity_1

identity_2

identity_3ИвC1/StatefulPartitionedCallвC3/StatefulPartitionedCallвF5/StatefulPartitionedCallвF6/StatefulPartitionedCallвexit_1/StatefulPartitionedCallвexit_2/StatefulPartitionedCallвexit_3/StatefulPartitionedCallвexit_4/StatefulPartitionedCall▀
C1/StatefulPartitionedCallStatefulPartitionedCallinputc1_80269c1_80271*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_C1_layer_call_and_return_conditional_losses_79795╙
S2/PartitionedCallPartitionedCall#C1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_S2_layer_call_and_return_conditional_losses_79762ї
C3/StatefulPartitionedCallStatefulPartitionedCallS2/PartitionedCall:output:0c3_80275c3_80277*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_C3_layer_call_and_return_conditional_losses_79813╙
S4/PartitionedCallPartitionedCall#C3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_S4_layer_call_and_return_conditional_losses_79774╬
Flatten/PartitionedCallPartitionedCallS4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79826╪
Flatten/PartitionedCall_1PartitionedCall#C3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79833╪
Flatten/PartitionedCall_2PartitionedCall#C1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79840Є
F5/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0f5_80284f5_80286*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_F5_layer_call_and_return_conditional_losses_79853ї
F6/StatefulPartitionedCallStatefulPartitionedCall#F5/StatefulPartitionedCall:output:0f6_80289f6_80291*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_F6_layer_call_and_return_conditional_losses_79870Е
exit_4/StatefulPartitionedCallStatefulPartitionedCall#F6/StatefulPartitionedCall:output:0exit_4_80294exit_4_80296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_4_layer_call_and_return_conditional_losses_79887Е
exit_3/StatefulPartitionedCallStatefulPartitionedCall#F5/StatefulPartitionedCall:output:0exit_3_80299exit_3_80301*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_3_layer_call_and_return_conditional_losses_79904Д
exit_2/StatefulPartitionedCallStatefulPartitionedCall"Flatten/PartitionedCall_1:output:0exit_2_80304exit_2_80306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_2_layer_call_and_return_conditional_losses_79921Д
exit_1/StatefulPartitionedCallStatefulPartitionedCall"Flatten/PartitionedCall_2:output:0exit_1_80309exit_1_80311*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_1_layer_call_and_return_conditional_losses_79938v
IdentityIdentity'exit_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_1Identity'exit_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_2Identity'exit_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_3Identity'exit_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
╛
NoOpNoOp^C1/StatefulPartitionedCall^C3/StatefulPartitionedCall^F5/StatefulPartitionedCall^F6/StatefulPartitionedCall^exit_1/StatefulPartitionedCall^exit_2/StatefulPartitionedCall^exit_3/StatefulPartitionedCall^exit_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 28
C1/StatefulPartitionedCallC1/StatefulPartitionedCall28
C3/StatefulPartitionedCallC3/StatefulPartitionedCall28
F5/StatefulPartitionedCallF5/StatefulPartitionedCall28
F6/StatefulPartitionedCallF6/StatefulPartitionedCall2@
exit_1/StatefulPartitionedCallexit_1/StatefulPartitionedCall2@
exit_2/StatefulPartitionedCallexit_2/StatefulPartitionedCall2@
exit_3/StatefulPartitionedCallexit_3/StatefulPartitionedCall2@
exit_4/StatefulPartitionedCallexit_4/StatefulPartitionedCall:V R
/
_output_shapes
:         

_user_specified_nameinput
¤

Ў
=__inference_C3_layer_call_and_return_conditional_losses_80699

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
с
╙
%__inference_model_layer_call_fn_80266	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	Аx
	unknown_4:x
	unknown_5:xT
	unknown_6:T
	unknown_7:T

	unknown_8:

	unknown_9:x


unknown_10:


unknown_11:	А


unknown_12:


unknown_13:	А


unknown_14:

identity

identity_1

identity_2

identity_3ИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:         
:         
:         
:         
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_80182o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         
q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         
q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:         

_user_specified_nameinput
─
^
B__inference_Flatten_layer_call_and_return_conditional_losses_80736

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
б

є
A__inference_exit_2_layer_call_and_return_conditional_losses_79921

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ш

я
=__inference_F5_layer_call_and_return_conditional_losses_80762

inputs1
matmul_readvariableop_resource:	Аx-
biasadd_readvariableop_resource:x
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         xP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         xa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         xw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
─
^
B__inference_Flatten_layer_call_and_return_conditional_losses_79833

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
─
^
B__inference_Flatten_layer_call_and_return_conditional_losses_80742

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
б

є
A__inference_exit_1_layer_call_and_return_conditional_losses_80802

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
О
Y
=__inference_S2_layer_call_and_return_conditional_losses_79762

inputs
identityл
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Я
>
"__inference_S4_layer_call_fn_80704

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_S4_layer_call_and_return_conditional_losses_79774Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
¤

Ў
=__inference_C3_layer_call_and_return_conditional_losses_79813

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┐
У
&__inference_exit_3_layer_call_fn_80831

inputs
unknown:x

	unknown_0:

identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_3_layer_call_and_return_conditional_losses_79904o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         x: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
Э

Є
A__inference_exit_4_layer_call_and_return_conditional_losses_79887

inputs0
matmul_readvariableop_resource:T
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
╖
П
"__inference_F6_layer_call_fn_80771

inputs
unknown:xT
	unknown_0:T
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_F6_layer_call_and_return_conditional_losses_79870o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         x: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
Э

Є
A__inference_exit_4_layer_call_and_return_conditional_losses_80862

inputs0
matmul_readvariableop_resource:T
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
┐
╤
#__inference_signature_wrapper_80421	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	Аx
	unknown_4:x
	unknown_5:xT
	unknown_6:T
	unknown_7:T

	unknown_8:

	unknown_9:x


unknown_10:


unknown_11:	А


unknown_12:


unknown_13:	А


unknown_14:

identity

identity_1

identity_2

identity_3ИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:         
:         
:         
:         
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_79753o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         
q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         
q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:         

_user_specified_nameinput
с
╙
%__inference_model_layer_call_fn_79989	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	Аx
	unknown_4:x
	unknown_5:xT
	unknown_6:T
	unknown_7:T

	unknown_8:

	unknown_9:x


unknown_10:


unknown_11:	А


unknown_12:


unknown_13:	А


unknown_14:

identity

identity_1

identity_2

identity_3ИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:         
:         
:         
:         
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_79948o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         
q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         
q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:         

_user_specified_nameinput
┐
У
&__inference_exit_4_layer_call_fn_80851

inputs
unknown:T

	unknown_0:

identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_4_layer_call_and_return_conditional_losses_79887o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         T: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
ф
╘
%__inference_model_layer_call_fn_80464

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	Аx
	unknown_4:x
	unknown_5:xT
	unknown_6:T
	unknown_7:T

	unknown_8:

	unknown_9:x


unknown_10:


unknown_11:	А


unknown_12:


unknown_13:	А


unknown_14:

identity

identity_1

identity_2

identity_3ИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:         
:         
:         
:         
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_79948o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         
q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         
q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ф

ю
=__inference_F6_layer_call_and_return_conditional_losses_79870

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Tr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         TP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         Ta
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         Tw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
ш8
┌
@__inference_model_layer_call_and_return_conditional_losses_80182

inputs"
c1_80133:
c1_80135:"
c3_80139:
c3_80141:
f5_80148:	Аx
f5_80150:x
f6_80153:xT
f6_80155:T
exit_4_80158:T

exit_4_80160:

exit_3_80163:x

exit_3_80165:

exit_2_80168:	А

exit_2_80170:

exit_1_80173:	А

exit_1_80175:

identity

identity_1

identity_2

identity_3ИвC1/StatefulPartitionedCallвC3/StatefulPartitionedCallвF5/StatefulPartitionedCallвF6/StatefulPartitionedCallвexit_1/StatefulPartitionedCallвexit_2/StatefulPartitionedCallвexit_3/StatefulPartitionedCallвexit_4/StatefulPartitionedCallр
C1/StatefulPartitionedCallStatefulPartitionedCallinputsc1_80133c1_80135*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_C1_layer_call_and_return_conditional_losses_79795╙
S2/PartitionedCallPartitionedCall#C1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_S2_layer_call_and_return_conditional_losses_79762ї
C3/StatefulPartitionedCallStatefulPartitionedCallS2/PartitionedCall:output:0c3_80139c3_80141*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_C3_layer_call_and_return_conditional_losses_79813╙
S4/PartitionedCallPartitionedCall#C3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_S4_layer_call_and_return_conditional_losses_79774╬
Flatten/PartitionedCallPartitionedCallS4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79826╪
Flatten/PartitionedCall_1PartitionedCall#C3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79833╪
Flatten/PartitionedCall_2PartitionedCall#C1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79840Є
F5/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0f5_80148f5_80150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_F5_layer_call_and_return_conditional_losses_79853ї
F6/StatefulPartitionedCallStatefulPartitionedCall#F5/StatefulPartitionedCall:output:0f6_80153f6_80155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_F6_layer_call_and_return_conditional_losses_79870Е
exit_4/StatefulPartitionedCallStatefulPartitionedCall#F6/StatefulPartitionedCall:output:0exit_4_80158exit_4_80160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_4_layer_call_and_return_conditional_losses_79887Е
exit_3/StatefulPartitionedCallStatefulPartitionedCall#F5/StatefulPartitionedCall:output:0exit_3_80163exit_3_80165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_3_layer_call_and_return_conditional_losses_79904Д
exit_2/StatefulPartitionedCallStatefulPartitionedCall"Flatten/PartitionedCall_1:output:0exit_2_80168exit_2_80170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_2_layer_call_and_return_conditional_losses_79921Д
exit_1/StatefulPartitionedCallStatefulPartitionedCall"Flatten/PartitionedCall_2:output:0exit_1_80173exit_1_80175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_1_layer_call_and_return_conditional_losses_79938v
IdentityIdentity'exit_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_1Identity'exit_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_2Identity'exit_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_3Identity'exit_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
╛
NoOpNoOp^C1/StatefulPartitionedCall^C3/StatefulPartitionedCall^F5/StatefulPartitionedCall^F6/StatefulPartitionedCall^exit_1/StatefulPartitionedCall^exit_2/StatefulPartitionedCall^exit_3/StatefulPartitionedCall^exit_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 28
C1/StatefulPartitionedCallC1/StatefulPartitionedCall28
C3/StatefulPartitionedCallC3/StatefulPartitionedCall28
F5/StatefulPartitionedCallF5/StatefulPartitionedCall28
F6/StatefulPartitionedCallF6/StatefulPartitionedCall2@
exit_1/StatefulPartitionedCallexit_1/StatefulPartitionedCall2@
exit_2/StatefulPartitionedCallexit_2/StatefulPartitionedCall2@
exit_3/StatefulPartitionedCallexit_3/StatefulPartitionedCall2@
exit_4/StatefulPartitionedCallexit_4/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
¤

Ў
=__inference_C1_layer_call_and_return_conditional_losses_80669

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ш

я
=__inference_F5_layer_call_and_return_conditional_losses_79853

inputs1
matmul_readvariableop_resource:	Аx-
biasadd_readvariableop_resource:x
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         xP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         xa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         xw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
БM
╥
@__inference_model_layer_call_and_return_conditional_losses_80578

inputs;
!c1_conv2d_readvariableop_resource:0
"c1_biasadd_readvariableop_resource:;
!c3_conv2d_readvariableop_resource:0
"c3_biasadd_readvariableop_resource:4
!f5_matmul_readvariableop_resource:	Аx0
"f5_biasadd_readvariableop_resource:x3
!f6_matmul_readvariableop_resource:xT0
"f6_biasadd_readvariableop_resource:T7
%exit_4_matmul_readvariableop_resource:T
4
&exit_4_biasadd_readvariableop_resource:
7
%exit_3_matmul_readvariableop_resource:x
4
&exit_3_biasadd_readvariableop_resource:
8
%exit_2_matmul_readvariableop_resource:	А
4
&exit_2_biasadd_readvariableop_resource:
8
%exit_1_matmul_readvariableop_resource:	А
4
&exit_1_biasadd_readvariableop_resource:

identity

identity_1

identity_2

identity_3ИвC1/BiasAdd/ReadVariableOpвC1/Conv2D/ReadVariableOpвC3/BiasAdd/ReadVariableOpвC3/Conv2D/ReadVariableOpвF5/BiasAdd/ReadVariableOpвF5/MatMul/ReadVariableOpвF6/BiasAdd/ReadVariableOpвF6/MatMul/ReadVariableOpвexit_1/BiasAdd/ReadVariableOpвexit_1/MatMul/ReadVariableOpвexit_2/BiasAdd/ReadVariableOpвexit_2/MatMul/ReadVariableOpвexit_3/BiasAdd/ReadVariableOpвexit_3/MatMul/ReadVariableOpвexit_4/BiasAdd/ReadVariableOpвexit_4/MatMul/ReadVariableOpВ
C1/Conv2D/ReadVariableOpReadVariableOp!c1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0а
	C1/Conv2DConv2Dinputs C1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
x
C1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

C1/BiasAddBiasAddC1/Conv2D:output:0!C1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^
C1/ReluReluC1/BiasAdd:output:0*
T0*/
_output_shapes
:         в

S2/AvgPoolAvgPoolC1/Relu:activations:0*
T0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
В
C3/Conv2D/ReadVariableOpReadVariableOp!c3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0н
	C3/Conv2DConv2DS2/AvgPool:output:0 C3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
x
C3/BiasAdd/ReadVariableOpReadVariableOp"c3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

C3/BiasAddBiasAddC3/Conv2D:output:0!C3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^
C3/ReluReluC3/BiasAdd:output:0*
T0*/
_output_shapes
:         в

S4/AvgPoolAvgPoolC3/Relu:activations:0*
T0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
^
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       z
Flatten/ReshapeReshapeS4/AvgPool:output:0Flatten/Const:output:0*
T0*(
_output_shapes
:         А`
Flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       А
Flatten/Reshape_1ReshapeC3/Relu:activations:0Flatten/Const_1:output:0*
T0*(
_output_shapes
:         А`
Flatten/Const_2Const*
_output_shapes
:*
dtype0*
valueB"    А  А
Flatten/Reshape_2ReshapeC1/Relu:activations:0Flatten/Const_2:output:0*
T0*(
_output_shapes
:         А{
F5/MatMul/ReadVariableOpReadVariableOp!f5_matmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype0Б
	F5/MatMulMatMulFlatten/Reshape:output:0 F5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         xx
F5/BiasAdd/ReadVariableOpReadVariableOp"f5_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0

F5/BiasAddBiasAddF5/MatMul:product:0!F5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         xV
F5/ReluReluF5/BiasAdd:output:0*
T0*'
_output_shapes
:         xz
F6/MatMul/ReadVariableOpReadVariableOp!f6_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0~
	F6/MatMulMatMulF5/Relu:activations:0 F6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Tx
F6/BiasAdd/ReadVariableOpReadVariableOp"f6_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0

F6/BiasAddBiasAddF6/MatMul:product:0!F6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         TV
F6/ReluReluF6/BiasAdd:output:0*
T0*'
_output_shapes
:         TВ
exit_4/MatMul/ReadVariableOpReadVariableOp%exit_4_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype0Ж
exit_4/MatMulMatMulF6/Relu:activations:0$exit_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
А
exit_4/BiasAdd/ReadVariableOpReadVariableOp&exit_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Л
exit_4/BiasAddBiasAddexit_4/MatMul:product:0%exit_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
exit_4/SoftmaxSoftmaxexit_4/BiasAdd:output:0*
T0*'
_output_shapes
:         
В
exit_3/MatMul/ReadVariableOpReadVariableOp%exit_3_matmul_readvariableop_resource*
_output_shapes

:x
*
dtype0Ж
exit_3/MatMulMatMulF5/Relu:activations:0$exit_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
А
exit_3/BiasAdd/ReadVariableOpReadVariableOp&exit_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Л
exit_3/BiasAddBiasAddexit_3/MatMul:product:0%exit_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
exit_3/SoftmaxSoftmaxexit_3/BiasAdd:output:0*
T0*'
_output_shapes
:         
Г
exit_2/MatMul/ReadVariableOpReadVariableOp%exit_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Л
exit_2/MatMulMatMulFlatten/Reshape_1:output:0$exit_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
А
exit_2/BiasAdd/ReadVariableOpReadVariableOp&exit_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Л
exit_2/BiasAddBiasAddexit_2/MatMul:product:0%exit_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
exit_2/SoftmaxSoftmaxexit_2/BiasAdd:output:0*
T0*'
_output_shapes
:         
Г
exit_1/MatMul/ReadVariableOpReadVariableOp%exit_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Л
exit_1/MatMulMatMulFlatten/Reshape_2:output:0$exit_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
А
exit_1/BiasAdd/ReadVariableOpReadVariableOp&exit_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Л
exit_1/BiasAddBiasAddexit_1/MatMul:product:0%exit_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
exit_1/SoftmaxSoftmaxexit_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
g
IdentityIdentityexit_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
i

Identity_1Identityexit_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
i

Identity_2Identityexit_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
i

Identity_3Identityexit_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Ю
NoOpNoOp^C1/BiasAdd/ReadVariableOp^C1/Conv2D/ReadVariableOp^C3/BiasAdd/ReadVariableOp^C3/Conv2D/ReadVariableOp^F5/BiasAdd/ReadVariableOp^F5/MatMul/ReadVariableOp^F6/BiasAdd/ReadVariableOp^F6/MatMul/ReadVariableOp^exit_1/BiasAdd/ReadVariableOp^exit_1/MatMul/ReadVariableOp^exit_2/BiasAdd/ReadVariableOp^exit_2/MatMul/ReadVariableOp^exit_3/BiasAdd/ReadVariableOp^exit_3/MatMul/ReadVariableOp^exit_4/BiasAdd/ReadVariableOp^exit_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 26
C1/BiasAdd/ReadVariableOpC1/BiasAdd/ReadVariableOp24
C1/Conv2D/ReadVariableOpC1/Conv2D/ReadVariableOp26
C3/BiasAdd/ReadVariableOpC3/BiasAdd/ReadVariableOp24
C3/Conv2D/ReadVariableOpC3/Conv2D/ReadVariableOp26
F5/BiasAdd/ReadVariableOpF5/BiasAdd/ReadVariableOp24
F5/MatMul/ReadVariableOpF5/MatMul/ReadVariableOp26
F6/BiasAdd/ReadVariableOpF6/BiasAdd/ReadVariableOp24
F6/MatMul/ReadVariableOpF6/MatMul/ReadVariableOp2>
exit_1/BiasAdd/ReadVariableOpexit_1/BiasAdd/ReadVariableOp2<
exit_1/MatMul/ReadVariableOpexit_1/MatMul/ReadVariableOp2>
exit_2/BiasAdd/ReadVariableOpexit_2/BiasAdd/ReadVariableOp2<
exit_2/MatMul/ReadVariableOpexit_2/MatMul/ReadVariableOp2>
exit_3/BiasAdd/ReadVariableOpexit_3/BiasAdd/ReadVariableOp2<
exit_3/MatMul/ReadVariableOpexit_3/MatMul/ReadVariableOp2>
exit_4/BiasAdd/ReadVariableOpexit_4/BiasAdd/ReadVariableOp2<
exit_4/MatMul/ReadVariableOpexit_4/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
О
Y
=__inference_S4_layer_call_and_return_conditional_losses_80709

inputs
identityл
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▀
Ч
"__inference_C1_layer_call_fn_80658

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_C1_layer_call_and_return_conditional_losses_79795w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ш8
┌
@__inference_model_layer_call_and_return_conditional_losses_79948

inputs"
c1_79796:
c1_79798:"
c3_79814:
c3_79816:
f5_79854:	Аx
f5_79856:x
f6_79871:xT
f6_79873:T
exit_4_79888:T

exit_4_79890:

exit_3_79905:x

exit_3_79907:

exit_2_79922:	А

exit_2_79924:

exit_1_79939:	А

exit_1_79941:

identity

identity_1

identity_2

identity_3ИвC1/StatefulPartitionedCallвC3/StatefulPartitionedCallвF5/StatefulPartitionedCallвF6/StatefulPartitionedCallвexit_1/StatefulPartitionedCallвexit_2/StatefulPartitionedCallвexit_3/StatefulPartitionedCallвexit_4/StatefulPartitionedCallр
C1/StatefulPartitionedCallStatefulPartitionedCallinputsc1_79796c1_79798*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_C1_layer_call_and_return_conditional_losses_79795╙
S2/PartitionedCallPartitionedCall#C1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_S2_layer_call_and_return_conditional_losses_79762ї
C3/StatefulPartitionedCallStatefulPartitionedCallS2/PartitionedCall:output:0c3_79814c3_79816*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_C3_layer_call_and_return_conditional_losses_79813╙
S4/PartitionedCallPartitionedCall#C3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_S4_layer_call_and_return_conditional_losses_79774╬
Flatten/PartitionedCallPartitionedCallS4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79826╪
Flatten/PartitionedCall_1PartitionedCall#C3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79833╪
Flatten/PartitionedCall_2PartitionedCall#C1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79840Є
F5/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0f5_79854f5_79856*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_F5_layer_call_and_return_conditional_losses_79853ї
F6/StatefulPartitionedCallStatefulPartitionedCall#F5/StatefulPartitionedCall:output:0f6_79871f6_79873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_F6_layer_call_and_return_conditional_losses_79870Е
exit_4/StatefulPartitionedCallStatefulPartitionedCall#F6/StatefulPartitionedCall:output:0exit_4_79888exit_4_79890*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_4_layer_call_and_return_conditional_losses_79887Е
exit_3/StatefulPartitionedCallStatefulPartitionedCall#F5/StatefulPartitionedCall:output:0exit_3_79905exit_3_79907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_3_layer_call_and_return_conditional_losses_79904Д
exit_2/StatefulPartitionedCallStatefulPartitionedCall"Flatten/PartitionedCall_1:output:0exit_2_79922exit_2_79924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_2_layer_call_and_return_conditional_losses_79921Д
exit_1/StatefulPartitionedCallStatefulPartitionedCall"Flatten/PartitionedCall_2:output:0exit_1_79939exit_1_79941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_1_layer_call_and_return_conditional_losses_79938v
IdentityIdentity'exit_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_1Identity'exit_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_2Identity'exit_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_3Identity'exit_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
╛
NoOpNoOp^C1/StatefulPartitionedCall^C3/StatefulPartitionedCall^F5/StatefulPartitionedCall^F6/StatefulPartitionedCall^exit_1/StatefulPartitionedCall^exit_2/StatefulPartitionedCall^exit_3/StatefulPartitionedCall^exit_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 28
C1/StatefulPartitionedCallC1/StatefulPartitionedCall28
C3/StatefulPartitionedCallC3/StatefulPartitionedCall28
F5/StatefulPartitionedCallF5/StatefulPartitionedCall28
F6/StatefulPartitionedCallF6/StatefulPartitionedCall2@
exit_1/StatefulPartitionedCallexit_1/StatefulPartitionedCall2@
exit_2/StatefulPartitionedCallexit_2/StatefulPartitionedCall2@
exit_3/StatefulPartitionedCallexit_3/StatefulPartitionedCall2@
exit_4/StatefulPartitionedCallexit_4/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
─
^
B__inference_Flatten_layer_call_and_return_conditional_losses_79840

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ўР
°'
!__inference__traced_restore_81324
file_prefix4
assignvariableop_c1_kernel:(
assignvariableop_1_c1_bias:6
assignvariableop_2_c3_kernel:(
assignvariableop_3_c3_bias:/
assignvariableop_4_f5_kernel:	Аx(
assignvariableop_5_f5_bias:x.
assignvariableop_6_f6_kernel:xT(
assignvariableop_7_f6_bias:T3
 assignvariableop_8_exit_1_kernel:	А
,
assignvariableop_9_exit_1_bias:
4
!assignvariableop_10_exit_2_kernel:	А
-
assignvariableop_11_exit_2_bias:
3
!assignvariableop_12_exit_3_kernel:x
-
assignvariableop_13_exit_3_bias:
3
!assignvariableop_14_exit_4_kernel:T
-
assignvariableop_15_exit_4_bias:
'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: %
assignvariableop_21_total_8: %
assignvariableop_22_count_8: %
assignvariableop_23_total_7: %
assignvariableop_24_count_7: %
assignvariableop_25_total_6: %
assignvariableop_26_count_6: %
assignvariableop_27_total_5: %
assignvariableop_28_count_5: %
assignvariableop_29_total_4: %
assignvariableop_30_count_4: %
assignvariableop_31_total_3: %
assignvariableop_32_count_3: %
assignvariableop_33_total_2: %
assignvariableop_34_count_2: %
assignvariableop_35_total_1: %
assignvariableop_36_count_1: #
assignvariableop_37_total: #
assignvariableop_38_count: >
$assignvariableop_39_adam_c1_kernel_m:0
"assignvariableop_40_adam_c1_bias_m:>
$assignvariableop_41_adam_c3_kernel_m:0
"assignvariableop_42_adam_c3_bias_m:7
$assignvariableop_43_adam_f5_kernel_m:	Аx0
"assignvariableop_44_adam_f5_bias_m:x6
$assignvariableop_45_adam_f6_kernel_m:xT0
"assignvariableop_46_adam_f6_bias_m:T;
(assignvariableop_47_adam_exit_1_kernel_m:	А
4
&assignvariableop_48_adam_exit_1_bias_m:
;
(assignvariableop_49_adam_exit_2_kernel_m:	А
4
&assignvariableop_50_adam_exit_2_bias_m:
:
(assignvariableop_51_adam_exit_3_kernel_m:x
4
&assignvariableop_52_adam_exit_3_bias_m:
:
(assignvariableop_53_adam_exit_4_kernel_m:T
4
&assignvariableop_54_adam_exit_4_bias_m:
>
$assignvariableop_55_adam_c1_kernel_v:0
"assignvariableop_56_adam_c1_bias_v:>
$assignvariableop_57_adam_c3_kernel_v:0
"assignvariableop_58_adam_c3_bias_v:7
$assignvariableop_59_adam_f5_kernel_v:	Аx0
"assignvariableop_60_adam_f5_bias_v:x6
$assignvariableop_61_adam_f6_kernel_v:xT0
"assignvariableop_62_adam_f6_bias_v:T;
(assignvariableop_63_adam_exit_1_kernel_v:	А
4
&assignvariableop_64_adam_exit_1_bias_v:
;
(assignvariableop_65_adam_exit_2_kernel_v:	А
4
&assignvariableop_66_adam_exit_2_bias_v:
:
(assignvariableop_67_adam_exit_3_kernel_v:x
4
&assignvariableop_68_adam_exit_3_bias_v:
:
(assignvariableop_69_adam_exit_4_kernel_v:T
4
&assignvariableop_70_adam_exit_4_bias_v:

identity_72ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_8вAssignVariableOp_9О&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*┤%
valueк%Bз%HB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHГ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*е
valueЫBШHB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Й
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╢
_output_shapesг
а::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*V
dtypesL
J2H	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOpAssignVariableOpassignvariableop_c1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_1AssignVariableOpassignvariableop_1_c1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_2AssignVariableOpassignvariableop_2_c3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_3AssignVariableOpassignvariableop_3_c3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_4AssignVariableOpassignvariableop_4_f5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_5AssignVariableOpassignvariableop_5_f5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_6AssignVariableOpassignvariableop_6_f6_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_7AssignVariableOpassignvariableop_7_f6_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_8AssignVariableOp assignvariableop_8_exit_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_exit_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_10AssignVariableOp!assignvariableop_10_exit_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_11AssignVariableOpassignvariableop_11_exit_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_12AssignVariableOp!assignvariableop_12_exit_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_13AssignVariableOpassignvariableop_13_exit_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_14AssignVariableOp!assignvariableop_14_exit_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_15AssignVariableOpassignvariableop_15_exit_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_8Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_8Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_7Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_7Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_6Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_6Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_5Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_5Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_4Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_4Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_3Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_3Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_2Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_39AssignVariableOp$assignvariableop_39_adam_c1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_40AssignVariableOp"assignvariableop_40_adam_c1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_41AssignVariableOp$assignvariableop_41_adam_c3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_42AssignVariableOp"assignvariableop_42_adam_c3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_43AssignVariableOp$assignvariableop_43_adam_f5_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_44AssignVariableOp"assignvariableop_44_adam_f5_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_45AssignVariableOp$assignvariableop_45_adam_f6_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_46AssignVariableOp"assignvariableop_46_adam_f6_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_exit_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_exit_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_exit_2_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_exit_2_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_exit_3_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_exit_3_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_exit_4_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_exit_4_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_55AssignVariableOp$assignvariableop_55_adam_c1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_56AssignVariableOp"assignvariableop_56_adam_c1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_57AssignVariableOp$assignvariableop_57_adam_c3_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_58AssignVariableOp"assignvariableop_58_adam_c3_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_59AssignVariableOp$assignvariableop_59_adam_f5_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_60AssignVariableOp"assignvariableop_60_adam_f5_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_61AssignVariableOp$assignvariableop_61_adam_f6_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_62AssignVariableOp"assignvariableop_62_adam_f6_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_exit_1_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_64AssignVariableOp&assignvariableop_64_adam_exit_1_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_exit_2_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_66AssignVariableOp&assignvariableop_66_adam_exit_2_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_exit_3_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_68AssignVariableOp&assignvariableop_68_adam_exit_3_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_exit_4_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_70AssignVariableOp&assignvariableop_70_adam_exit_4_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 щ
Identity_71Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_72IdentityIdentity_71:output:0^NoOp_1*
T0*
_output_shapes
: ╓
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_72Identity_72:output:0*е
_input_shapesУ
Р: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Я
>
"__inference_S2_layer_call_fn_80674

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_S2_layer_call_and_return_conditional_losses_79762Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
БM
╥
@__inference_model_layer_call_and_return_conditional_losses_80649

inputs;
!c1_conv2d_readvariableop_resource:0
"c1_biasadd_readvariableop_resource:;
!c3_conv2d_readvariableop_resource:0
"c3_biasadd_readvariableop_resource:4
!f5_matmul_readvariableop_resource:	Аx0
"f5_biasadd_readvariableop_resource:x3
!f6_matmul_readvariableop_resource:xT0
"f6_biasadd_readvariableop_resource:T7
%exit_4_matmul_readvariableop_resource:T
4
&exit_4_biasadd_readvariableop_resource:
7
%exit_3_matmul_readvariableop_resource:x
4
&exit_3_biasadd_readvariableop_resource:
8
%exit_2_matmul_readvariableop_resource:	А
4
&exit_2_biasadd_readvariableop_resource:
8
%exit_1_matmul_readvariableop_resource:	А
4
&exit_1_biasadd_readvariableop_resource:

identity

identity_1

identity_2

identity_3ИвC1/BiasAdd/ReadVariableOpвC1/Conv2D/ReadVariableOpвC3/BiasAdd/ReadVariableOpвC3/Conv2D/ReadVariableOpвF5/BiasAdd/ReadVariableOpвF5/MatMul/ReadVariableOpвF6/BiasAdd/ReadVariableOpвF6/MatMul/ReadVariableOpвexit_1/BiasAdd/ReadVariableOpвexit_1/MatMul/ReadVariableOpвexit_2/BiasAdd/ReadVariableOpвexit_2/MatMul/ReadVariableOpвexit_3/BiasAdd/ReadVariableOpвexit_3/MatMul/ReadVariableOpвexit_4/BiasAdd/ReadVariableOpвexit_4/MatMul/ReadVariableOpВ
C1/Conv2D/ReadVariableOpReadVariableOp!c1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0а
	C1/Conv2DConv2Dinputs C1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
x
C1/BiasAdd/ReadVariableOpReadVariableOp"c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

C1/BiasAddBiasAddC1/Conv2D:output:0!C1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^
C1/ReluReluC1/BiasAdd:output:0*
T0*/
_output_shapes
:         в

S2/AvgPoolAvgPoolC1/Relu:activations:0*
T0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
В
C3/Conv2D/ReadVariableOpReadVariableOp!c3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0н
	C3/Conv2DConv2DS2/AvgPool:output:0 C3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
x
C3/BiasAdd/ReadVariableOpReadVariableOp"c3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж

C3/BiasAddBiasAddC3/Conv2D:output:0!C3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^
C3/ReluReluC3/BiasAdd:output:0*
T0*/
_output_shapes
:         в

S4/AvgPoolAvgPoolC3/Relu:activations:0*
T0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
^
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       z
Flatten/ReshapeReshapeS4/AvgPool:output:0Flatten/Const:output:0*
T0*(
_output_shapes
:         А`
Flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       А
Flatten/Reshape_1ReshapeC3/Relu:activations:0Flatten/Const_1:output:0*
T0*(
_output_shapes
:         А`
Flatten/Const_2Const*
_output_shapes
:*
dtype0*
valueB"    А  А
Flatten/Reshape_2ReshapeC1/Relu:activations:0Flatten/Const_2:output:0*
T0*(
_output_shapes
:         А{
F5/MatMul/ReadVariableOpReadVariableOp!f5_matmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype0Б
	F5/MatMulMatMulFlatten/Reshape:output:0 F5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         xx
F5/BiasAdd/ReadVariableOpReadVariableOp"f5_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0

F5/BiasAddBiasAddF5/MatMul:product:0!F5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         xV
F5/ReluReluF5/BiasAdd:output:0*
T0*'
_output_shapes
:         xz
F6/MatMul/ReadVariableOpReadVariableOp!f6_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0~
	F6/MatMulMatMulF5/Relu:activations:0 F6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Tx
F6/BiasAdd/ReadVariableOpReadVariableOp"f6_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0

F6/BiasAddBiasAddF6/MatMul:product:0!F6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         TV
F6/ReluReluF6/BiasAdd:output:0*
T0*'
_output_shapes
:         TВ
exit_4/MatMul/ReadVariableOpReadVariableOp%exit_4_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype0Ж
exit_4/MatMulMatMulF6/Relu:activations:0$exit_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
А
exit_4/BiasAdd/ReadVariableOpReadVariableOp&exit_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Л
exit_4/BiasAddBiasAddexit_4/MatMul:product:0%exit_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
exit_4/SoftmaxSoftmaxexit_4/BiasAdd:output:0*
T0*'
_output_shapes
:         
В
exit_3/MatMul/ReadVariableOpReadVariableOp%exit_3_matmul_readvariableop_resource*
_output_shapes

:x
*
dtype0Ж
exit_3/MatMulMatMulF5/Relu:activations:0$exit_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
А
exit_3/BiasAdd/ReadVariableOpReadVariableOp&exit_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Л
exit_3/BiasAddBiasAddexit_3/MatMul:product:0%exit_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
exit_3/SoftmaxSoftmaxexit_3/BiasAdd:output:0*
T0*'
_output_shapes
:         
Г
exit_2/MatMul/ReadVariableOpReadVariableOp%exit_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Л
exit_2/MatMulMatMulFlatten/Reshape_1:output:0$exit_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
А
exit_2/BiasAdd/ReadVariableOpReadVariableOp&exit_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Л
exit_2/BiasAddBiasAddexit_2/MatMul:product:0%exit_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
exit_2/SoftmaxSoftmaxexit_2/BiasAdd:output:0*
T0*'
_output_shapes
:         
Г
exit_1/MatMul/ReadVariableOpReadVariableOp%exit_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Л
exit_1/MatMulMatMulFlatten/Reshape_2:output:0$exit_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
А
exit_1/BiasAdd/ReadVariableOpReadVariableOp&exit_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Л
exit_1/BiasAddBiasAddexit_1/MatMul:product:0%exit_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
exit_1/SoftmaxSoftmaxexit_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
g
IdentityIdentityexit_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
i

Identity_1Identityexit_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
i

Identity_2Identityexit_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
i

Identity_3Identityexit_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Ю
NoOpNoOp^C1/BiasAdd/ReadVariableOp^C1/Conv2D/ReadVariableOp^C3/BiasAdd/ReadVariableOp^C3/Conv2D/ReadVariableOp^F5/BiasAdd/ReadVariableOp^F5/MatMul/ReadVariableOp^F6/BiasAdd/ReadVariableOp^F6/MatMul/ReadVariableOp^exit_1/BiasAdd/ReadVariableOp^exit_1/MatMul/ReadVariableOp^exit_2/BiasAdd/ReadVariableOp^exit_2/MatMul/ReadVariableOp^exit_3/BiasAdd/ReadVariableOp^exit_3/MatMul/ReadVariableOp^exit_4/BiasAdd/ReadVariableOp^exit_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 26
C1/BiasAdd/ReadVariableOpC1/BiasAdd/ReadVariableOp24
C1/Conv2D/ReadVariableOpC1/Conv2D/ReadVariableOp26
C3/BiasAdd/ReadVariableOpC3/BiasAdd/ReadVariableOp24
C3/Conv2D/ReadVariableOpC3/Conv2D/ReadVariableOp26
F5/BiasAdd/ReadVariableOpF5/BiasAdd/ReadVariableOp24
F5/MatMul/ReadVariableOpF5/MatMul/ReadVariableOp26
F6/BiasAdd/ReadVariableOpF6/BiasAdd/ReadVariableOp24
F6/MatMul/ReadVariableOpF6/MatMul/ReadVariableOp2>
exit_1/BiasAdd/ReadVariableOpexit_1/BiasAdd/ReadVariableOp2<
exit_1/MatMul/ReadVariableOpexit_1/MatMul/ReadVariableOp2>
exit_2/BiasAdd/ReadVariableOpexit_2/BiasAdd/ReadVariableOp2<
exit_2/MatMul/ReadVariableOpexit_2/MatMul/ReadVariableOp2>
exit_3/BiasAdd/ReadVariableOpexit_3/BiasAdd/ReadVariableOp2<
exit_3/MatMul/ReadVariableOpexit_3/MatMul/ReadVariableOp2>
exit_4/BiasAdd/ReadVariableOpexit_4/BiasAdd/ReadVariableOp2<
exit_4/MatMul/ReadVariableOpexit_4/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
║
Р
"__inference_F5_layer_call_fn_80751

inputs
unknown:	Аx
	unknown_0:x
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_F5_layer_call_and_return_conditional_losses_79853o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         x`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ф
╘
%__inference_model_layer_call_fn_80507

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	Аx
	unknown_4:x
	unknown_5:xT
	unknown_6:T
	unknown_7:T

	unknown_8:

	unknown_9:x


unknown_10:


unknown_11:	А


unknown_12:


unknown_13:	А


unknown_14:

identity

identity_1

identity_2

identity_3ИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:         
:         
:         
:         
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_80182o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         
q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         
q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
─
^
B__inference_Flatten_layer_call_and_return_conditional_losses_80730

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ь
З
__inference__traced_save_81101
file_prefix(
$savev2_c1_kernel_read_readvariableop&
"savev2_c1_bias_read_readvariableop(
$savev2_c3_kernel_read_readvariableop&
"savev2_c3_bias_read_readvariableop(
$savev2_f5_kernel_read_readvariableop&
"savev2_f5_bias_read_readvariableop(
$savev2_f6_kernel_read_readvariableop&
"savev2_f6_bias_read_readvariableop,
(savev2_exit_1_kernel_read_readvariableop*
&savev2_exit_1_bias_read_readvariableop,
(savev2_exit_2_kernel_read_readvariableop*
&savev2_exit_2_bias_read_readvariableop,
(savev2_exit_3_kernel_read_readvariableop*
&savev2_exit_3_bias_read_readvariableop,
(savev2_exit_4_kernel_read_readvariableop*
&savev2_exit_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_8_read_readvariableop&
"savev2_count_8_read_readvariableop&
"savev2_total_7_read_readvariableop&
"savev2_count_7_read_readvariableop&
"savev2_total_6_read_readvariableop&
"savev2_count_6_read_readvariableop&
"savev2_total_5_read_readvariableop&
"savev2_count_5_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_adam_c1_kernel_m_read_readvariableop-
)savev2_adam_c1_bias_m_read_readvariableop/
+savev2_adam_c3_kernel_m_read_readvariableop-
)savev2_adam_c3_bias_m_read_readvariableop/
+savev2_adam_f5_kernel_m_read_readvariableop-
)savev2_adam_f5_bias_m_read_readvariableop/
+savev2_adam_f6_kernel_m_read_readvariableop-
)savev2_adam_f6_bias_m_read_readvariableop3
/savev2_adam_exit_1_kernel_m_read_readvariableop1
-savev2_adam_exit_1_bias_m_read_readvariableop3
/savev2_adam_exit_2_kernel_m_read_readvariableop1
-savev2_adam_exit_2_bias_m_read_readvariableop3
/savev2_adam_exit_3_kernel_m_read_readvariableop1
-savev2_adam_exit_3_bias_m_read_readvariableop3
/savev2_adam_exit_4_kernel_m_read_readvariableop1
-savev2_adam_exit_4_bias_m_read_readvariableop/
+savev2_adam_c1_kernel_v_read_readvariableop-
)savev2_adam_c1_bias_v_read_readvariableop/
+savev2_adam_c3_kernel_v_read_readvariableop-
)savev2_adam_c3_bias_v_read_readvariableop/
+savev2_adam_f5_kernel_v_read_readvariableop-
)savev2_adam_f5_bias_v_read_readvariableop/
+savev2_adam_f6_kernel_v_read_readvariableop-
)savev2_adam_f6_bias_v_read_readvariableop3
/savev2_adam_exit_1_kernel_v_read_readvariableop1
-savev2_adam_exit_1_bias_v_read_readvariableop3
/savev2_adam_exit_2_kernel_v_read_readvariableop1
-savev2_adam_exit_2_bias_v_read_readvariableop3
/savev2_adam_exit_3_kernel_v_read_readvariableop1
-savev2_adam_exit_3_bias_v_read_readvariableop3
/savev2_adam_exit_4_kernel_v_read_readvariableop1
-savev2_adam_exit_4_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Л&
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*┤%
valueк%Bз%HB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHА
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*е
valueЫBШHB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ў
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_c1_kernel_read_readvariableop"savev2_c1_bias_read_readvariableop$savev2_c3_kernel_read_readvariableop"savev2_c3_bias_read_readvariableop$savev2_f5_kernel_read_readvariableop"savev2_f5_bias_read_readvariableop$savev2_f6_kernel_read_readvariableop"savev2_f6_bias_read_readvariableop(savev2_exit_1_kernel_read_readvariableop&savev2_exit_1_bias_read_readvariableop(savev2_exit_2_kernel_read_readvariableop&savev2_exit_2_bias_read_readvariableop(savev2_exit_3_kernel_read_readvariableop&savev2_exit_3_bias_read_readvariableop(savev2_exit_4_kernel_read_readvariableop&savev2_exit_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_8_read_readvariableop"savev2_count_8_read_readvariableop"savev2_total_7_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_6_read_readvariableop"savev2_count_6_read_readvariableop"savev2_total_5_read_readvariableop"savev2_count_5_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_adam_c1_kernel_m_read_readvariableop)savev2_adam_c1_bias_m_read_readvariableop+savev2_adam_c3_kernel_m_read_readvariableop)savev2_adam_c3_bias_m_read_readvariableop+savev2_adam_f5_kernel_m_read_readvariableop)savev2_adam_f5_bias_m_read_readvariableop+savev2_adam_f6_kernel_m_read_readvariableop)savev2_adam_f6_bias_m_read_readvariableop/savev2_adam_exit_1_kernel_m_read_readvariableop-savev2_adam_exit_1_bias_m_read_readvariableop/savev2_adam_exit_2_kernel_m_read_readvariableop-savev2_adam_exit_2_bias_m_read_readvariableop/savev2_adam_exit_3_kernel_m_read_readvariableop-savev2_adam_exit_3_bias_m_read_readvariableop/savev2_adam_exit_4_kernel_m_read_readvariableop-savev2_adam_exit_4_bias_m_read_readvariableop+savev2_adam_c1_kernel_v_read_readvariableop)savev2_adam_c1_bias_v_read_readvariableop+savev2_adam_c3_kernel_v_read_readvariableop)savev2_adam_c3_bias_v_read_readvariableop+savev2_adam_f5_kernel_v_read_readvariableop)savev2_adam_f5_bias_v_read_readvariableop+savev2_adam_f6_kernel_v_read_readvariableop)savev2_adam_f6_bias_v_read_readvariableop/savev2_adam_exit_1_kernel_v_read_readvariableop-savev2_adam_exit_1_bias_v_read_readvariableop/savev2_adam_exit_2_kernel_v_read_readvariableop-savev2_adam_exit_2_bias_v_read_readvariableop/savev2_adam_exit_3_kernel_v_read_readvariableop-savev2_adam_exit_3_bias_v_read_readvariableop/savev2_adam_exit_4_kernel_v_read_readvariableop-savev2_adam_exit_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *V
dtypesL
J2H	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*А
_input_shapesю
ы: :::::	Аx:x:xT:T:	А
:
:	А
:
:x
:
:T
:
: : : : : : : : : : : : : : : : : : : : : : : :::::	Аx:x:xT:T:	А
:
:	А
:
:x
:
:T
:
:::::	Аx:x:xT:T:	А
:
:	А
:
:x
:
:T
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	Аx: 

_output_shapes
:x:$ 

_output_shapes

:xT: 

_output_shapes
:T:%	!

_output_shapes
:	А
: 


_output_shapes
:
:%!

_output_shapes
:	А
: 

_output_shapes
:
:$ 

_output_shapes

:x
: 

_output_shapes
:
:$ 

_output_shapes

:T
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::%,!

_output_shapes
:	Аx: -

_output_shapes
:x:$. 

_output_shapes

:xT: /

_output_shapes
:T:%0!

_output_shapes
:	А
: 1

_output_shapes
:
:%2!

_output_shapes
:	А
: 3

_output_shapes
:
:$4 

_output_shapes

:x
: 5

_output_shapes
:
:$6 

_output_shapes

:T
: 7

_output_shapes
:
:,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::%<!

_output_shapes
:	Аx: =

_output_shapes
:x:$> 

_output_shapes

:xT: ?

_output_shapes
:T:%@!

_output_shapes
:	А
: A

_output_shapes
:
:%B!

_output_shapes
:	А
: C

_output_shapes
:
:$D 

_output_shapes

:x
: E

_output_shapes
:
:$F 

_output_shapes

:T
: G

_output_shapes
:
:H

_output_shapes
: 
·U
ё
 __inference__wrapped_model_79753	
inputA
'model_c1_conv2d_readvariableop_resource:6
(model_c1_biasadd_readvariableop_resource:A
'model_c3_conv2d_readvariableop_resource:6
(model_c3_biasadd_readvariableop_resource::
'model_f5_matmul_readvariableop_resource:	Аx6
(model_f5_biasadd_readvariableop_resource:x9
'model_f6_matmul_readvariableop_resource:xT6
(model_f6_biasadd_readvariableop_resource:T=
+model_exit_4_matmul_readvariableop_resource:T
:
,model_exit_4_biasadd_readvariableop_resource:
=
+model_exit_3_matmul_readvariableop_resource:x
:
,model_exit_3_biasadd_readvariableop_resource:
>
+model_exit_2_matmul_readvariableop_resource:	А
:
,model_exit_2_biasadd_readvariableop_resource:
>
+model_exit_1_matmul_readvariableop_resource:	А
:
,model_exit_1_biasadd_readvariableop_resource:

identity

identity_1

identity_2

identity_3Ивmodel/C1/BiasAdd/ReadVariableOpвmodel/C1/Conv2D/ReadVariableOpвmodel/C3/BiasAdd/ReadVariableOpвmodel/C3/Conv2D/ReadVariableOpвmodel/F5/BiasAdd/ReadVariableOpвmodel/F5/MatMul/ReadVariableOpвmodel/F6/BiasAdd/ReadVariableOpвmodel/F6/MatMul/ReadVariableOpв#model/exit_1/BiasAdd/ReadVariableOpв"model/exit_1/MatMul/ReadVariableOpв#model/exit_2/BiasAdd/ReadVariableOpв"model/exit_2/MatMul/ReadVariableOpв#model/exit_3/BiasAdd/ReadVariableOpв"model/exit_3/MatMul/ReadVariableOpв#model/exit_4/BiasAdd/ReadVariableOpв"model/exit_4/MatMul/ReadVariableOpО
model/C1/Conv2D/ReadVariableOpReadVariableOp'model_c1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0л
model/C1/Conv2DConv2Dinput&model/C1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
Д
model/C1/BiasAdd/ReadVariableOpReadVariableOp(model_c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ш
model/C1/BiasAddBiasAddmodel/C1/Conv2D:output:0'model/C1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         j
model/C1/ReluRelumodel/C1/BiasAdd:output:0*
T0*/
_output_shapes
:         о
model/S2/AvgPoolAvgPoolmodel/C1/Relu:activations:0*
T0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
О
model/C3/Conv2D/ReadVariableOpReadVariableOp'model_c3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0┐
model/C3/Conv2DConv2Dmodel/S2/AvgPool:output:0&model/C3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
Д
model/C3/BiasAdd/ReadVariableOpReadVariableOp(model_c3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ш
model/C3/BiasAddBiasAddmodel/C3/Conv2D:output:0'model/C3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         j
model/C3/ReluRelumodel/C3/BiasAdd:output:0*
T0*/
_output_shapes
:         о
model/S4/AvgPoolAvgPoolmodel/C3/Relu:activations:0*
T0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
d
model/Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       М
model/Flatten/ReshapeReshapemodel/S4/AvgPool:output:0model/Flatten/Const:output:0*
T0*(
_output_shapes
:         Аf
model/Flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Т
model/Flatten/Reshape_1Reshapemodel/C3/Relu:activations:0model/Flatten/Const_1:output:0*
T0*(
_output_shapes
:         Аf
model/Flatten/Const_2Const*
_output_shapes
:*
dtype0*
valueB"    А  Т
model/Flatten/Reshape_2Reshapemodel/C1/Relu:activations:0model/Flatten/Const_2:output:0*
T0*(
_output_shapes
:         АЗ
model/F5/MatMul/ReadVariableOpReadVariableOp'model_f5_matmul_readvariableop_resource*
_output_shapes
:	Аx*
dtype0У
model/F5/MatMulMatMulmodel/Flatten/Reshape:output:0&model/F5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         xД
model/F5/BiasAdd/ReadVariableOpReadVariableOp(model_f5_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0С
model/F5/BiasAddBiasAddmodel/F5/MatMul:product:0'model/F5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         xb
model/F5/ReluRelumodel/F5/BiasAdd:output:0*
T0*'
_output_shapes
:         xЖ
model/F6/MatMul/ReadVariableOpReadVariableOp'model_f6_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0Р
model/F6/MatMulMatMulmodel/F5/Relu:activations:0&model/F6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         TД
model/F6/BiasAdd/ReadVariableOpReadVariableOp(model_f6_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0С
model/F6/BiasAddBiasAddmodel/F6/MatMul:product:0'model/F6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Tb
model/F6/ReluRelumodel/F6/BiasAdd:output:0*
T0*'
_output_shapes
:         TО
"model/exit_4/MatMul/ReadVariableOpReadVariableOp+model_exit_4_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype0Ш
model/exit_4/MatMulMatMulmodel/F6/Relu:activations:0*model/exit_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
М
#model/exit_4/BiasAdd/ReadVariableOpReadVariableOp,model_exit_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Э
model/exit_4/BiasAddBiasAddmodel/exit_4/MatMul:product:0+model/exit_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
p
model/exit_4/SoftmaxSoftmaxmodel/exit_4/BiasAdd:output:0*
T0*'
_output_shapes
:         
О
"model/exit_3/MatMul/ReadVariableOpReadVariableOp+model_exit_3_matmul_readvariableop_resource*
_output_shapes

:x
*
dtype0Ш
model/exit_3/MatMulMatMulmodel/F5/Relu:activations:0*model/exit_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
М
#model/exit_3/BiasAdd/ReadVariableOpReadVariableOp,model_exit_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Э
model/exit_3/BiasAddBiasAddmodel/exit_3/MatMul:product:0+model/exit_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
p
model/exit_3/SoftmaxSoftmaxmodel/exit_3/BiasAdd:output:0*
T0*'
_output_shapes
:         
П
"model/exit_2/MatMul/ReadVariableOpReadVariableOp+model_exit_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Э
model/exit_2/MatMulMatMul model/Flatten/Reshape_1:output:0*model/exit_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
М
#model/exit_2/BiasAdd/ReadVariableOpReadVariableOp,model_exit_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Э
model/exit_2/BiasAddBiasAddmodel/exit_2/MatMul:product:0+model/exit_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
p
model/exit_2/SoftmaxSoftmaxmodel/exit_2/BiasAdd:output:0*
T0*'
_output_shapes
:         
П
"model/exit_1/MatMul/ReadVariableOpReadVariableOp+model_exit_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Э
model/exit_1/MatMulMatMul model/Flatten/Reshape_2:output:0*model/exit_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
М
#model/exit_1/BiasAdd/ReadVariableOpReadVariableOp,model_exit_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Э
model/exit_1/BiasAddBiasAddmodel/exit_1/MatMul:product:0+model/exit_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
p
model/exit_1/SoftmaxSoftmaxmodel/exit_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
m
IdentityIdentitymodel/exit_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
o

Identity_1Identitymodel/exit_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
o

Identity_2Identitymodel/exit_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
o

Identity_3Identitymodel/exit_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
■
NoOpNoOp ^model/C1/BiasAdd/ReadVariableOp^model/C1/Conv2D/ReadVariableOp ^model/C3/BiasAdd/ReadVariableOp^model/C3/Conv2D/ReadVariableOp ^model/F5/BiasAdd/ReadVariableOp^model/F5/MatMul/ReadVariableOp ^model/F6/BiasAdd/ReadVariableOp^model/F6/MatMul/ReadVariableOp$^model/exit_1/BiasAdd/ReadVariableOp#^model/exit_1/MatMul/ReadVariableOp$^model/exit_2/BiasAdd/ReadVariableOp#^model/exit_2/MatMul/ReadVariableOp$^model/exit_3/BiasAdd/ReadVariableOp#^model/exit_3/MatMul/ReadVariableOp$^model/exit_4/BiasAdd/ReadVariableOp#^model/exit_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2B
model/C1/BiasAdd/ReadVariableOpmodel/C1/BiasAdd/ReadVariableOp2@
model/C1/Conv2D/ReadVariableOpmodel/C1/Conv2D/ReadVariableOp2B
model/C3/BiasAdd/ReadVariableOpmodel/C3/BiasAdd/ReadVariableOp2@
model/C3/Conv2D/ReadVariableOpmodel/C3/Conv2D/ReadVariableOp2B
model/F5/BiasAdd/ReadVariableOpmodel/F5/BiasAdd/ReadVariableOp2@
model/F5/MatMul/ReadVariableOpmodel/F5/MatMul/ReadVariableOp2B
model/F6/BiasAdd/ReadVariableOpmodel/F6/BiasAdd/ReadVariableOp2@
model/F6/MatMul/ReadVariableOpmodel/F6/MatMul/ReadVariableOp2J
#model/exit_1/BiasAdd/ReadVariableOp#model/exit_1/BiasAdd/ReadVariableOp2H
"model/exit_1/MatMul/ReadVariableOp"model/exit_1/MatMul/ReadVariableOp2J
#model/exit_2/BiasAdd/ReadVariableOp#model/exit_2/BiasAdd/ReadVariableOp2H
"model/exit_2/MatMul/ReadVariableOp"model/exit_2/MatMul/ReadVariableOp2J
#model/exit_3/BiasAdd/ReadVariableOp#model/exit_3/BiasAdd/ReadVariableOp2H
"model/exit_3/MatMul/ReadVariableOp"model/exit_3/MatMul/ReadVariableOp2J
#model/exit_4/BiasAdd/ReadVariableOp#model/exit_4/BiasAdd/ReadVariableOp2H
"model/exit_4/MatMul/ReadVariableOp"model/exit_4/MatMul/ReadVariableOp:V R
/
_output_shapes
:         

_user_specified_nameinput
▀
Ч
"__inference_C3_layer_call_fn_80688

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__inference_C3_layer_call_and_return_conditional_losses_79813w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
б

є
A__inference_exit_1_layer_call_and_return_conditional_losses_79938

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
о
C
'__inference_Flatten_layer_call_fn_80724

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79826a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
о
C
'__inference_Flatten_layer_call_fn_80719

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79833a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┬
Ф
&__inference_exit_2_layer_call_fn_80811

inputs
unknown:	А

	unknown_0:

identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_exit_2_layer_call_and_return_conditional_losses_79921o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Э

Є
A__inference_exit_3_layer_call_and_return_conditional_losses_79904

inputs0
matmul_readvariableop_resource:x
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
о
C
'__inference_Flatten_layer_call_fn_80714

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_79840a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs"┐L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*с
serving_default═
?
input6
serving_default_input:0         :
exit_10
StatefulPartitionedCall:0         
:
exit_20
StatefulPartitionedCall:1         
:
exit_30
StatefulPartitionedCall:2         
:
exit_40
StatefulPartitionedCall:3         
tensorflow/serving/predict:ог
╤
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
▌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
е
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op"
_tf_keras_layer
е
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
е
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
╗
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias"
_tf_keras_layer
╗
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

Pkernel
Qbias"
_tf_keras_layer
╗
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias"
_tf_keras_layer
╗
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias"
_tf_keras_layer
╗
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias"
_tf_keras_layer
Ц
0
1
+2
,3
@4
A5
H6
I7
P8
Q9
X10
Y11
`12
a13
h14
i15"
trackable_list_wrapper
Ц
0
1
+2
,3
@4
A5
H6
I7
P8
Q9
X10
Y11
`12
a13
h14
i15"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╩
otrace_0
ptrace_1
qtrace_2
rtrace_32▀
%__inference_model_layer_call_fn_79989
%__inference_model_layer_call_fn_80464
%__inference_model_layer_call_fn_80507
%__inference_model_layer_call_fn_80266└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zotrace_0zptrace_1zqtrace_2zrtrace_3
╢
strace_0
ttrace_1
utrace_2
vtrace_32╦
@__inference_model_layer_call_and_return_conditional_losses_80578
@__inference_model_layer_call_and_return_conditional_losses_80649
@__inference_model_layer_call_and_return_conditional_losses_80318
@__inference_model_layer_call_and_return_conditional_losses_80370└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zstrace_0zttrace_1zutrace_2zvtrace_3
╔B╞
 __inference__wrapped_model_79753input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
У
witer

xbeta_1

ybeta_2
	zdecay
{learning_ratem mА+mБ,mВ@mГAmДHmЕImЖPmЗQmИXmЙYmК`mЛamМhmНimОvПvР+vС,vТ@vУAvФHvХIvЦPvЧQvШXvЩYvЪ`vЫavЬhvЭivЮ"
	optimizer
,
|serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
п
}non_trainable_variables

~layers
metrics
 Аlayer_regularization_losses
Бlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ш
Вtrace_02╔
"__inference_C1_layer_call_fn_80658в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zВtrace_0
Г
Гtrace_02ф
=__inference_C1_layer_call_and_return_conditional_losses_80669в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0
#:!2	C1/kernel
:2C1/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
ш
Йtrace_02╔
"__inference_S2_layer_call_fn_80674в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЙtrace_0
Г
Кtrace_02ф
=__inference_S2_layer_call_and_return_conditional_losses_80679в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zКtrace_0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ш
Рtrace_02╔
"__inference_C3_layer_call_fn_80688в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zРtrace_0
Г
Сtrace_02ф
=__inference_C3_layer_call_and_return_conditional_losses_80699в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zСtrace_0
#:!2	C3/kernel
:2C3/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ш
Чtrace_02╔
"__inference_S4_layer_call_fn_80704в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЧtrace_0
Г
Шtrace_02ф
=__inference_S4_layer_call_and_return_conditional_losses_80709в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zШtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
ў
Юtrace_0
Яtrace_1
аtrace_22а
'__inference_Flatten_layer_call_fn_80714
'__inference_Flatten_layer_call_fn_80719
'__inference_Flatten_layer_call_fn_80724в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЮtrace_0zЯtrace_1zаtrace_2
╚
бtrace_0
вtrace_1
гtrace_22ё
B__inference_Flatten_layer_call_and_return_conditional_losses_80730
B__inference_Flatten_layer_call_and_return_conditional_losses_80736
B__inference_Flatten_layer_call_and_return_conditional_losses_80742в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0zвtrace_1zгtrace_2
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
ш
йtrace_02╔
"__inference_F5_layer_call_fn_80751в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zйtrace_0
Г
кtrace_02ф
=__inference_F5_layer_call_and_return_conditional_losses_80762в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zкtrace_0
:	Аx2	F5/kernel
:x2F5/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ш
░trace_02╔
"__inference_F6_layer_call_fn_80771в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z░trace_0
Г
▒trace_02ф
=__inference_F6_layer_call_and_return_conditional_losses_80782в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▒trace_0
:xT2	F6/kernel
:T2F6/bias
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▓non_trainable_variables
│layers
┤metrics
 ╡layer_regularization_losses
╢layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ь
╖trace_02═
&__inference_exit_1_layer_call_fn_80791в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╖trace_0
З
╕trace_02ш
A__inference_exit_1_layer_call_and_return_conditional_losses_80802в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╕trace_0
 :	А
2exit_1/kernel
:
2exit_1/bias
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
ь
╛trace_02═
&__inference_exit_2_layer_call_fn_80811в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╛trace_0
З
┐trace_02ш
A__inference_exit_2_layer_call_and_return_conditional_losses_80822в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┐trace_0
 :	А
2exit_2/kernel
:
2exit_2/bias
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
ь
┼trace_02═
&__inference_exit_3_layer_call_fn_80831в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┼trace_0
З
╞trace_02ш
A__inference_exit_3_layer_call_and_return_conditional_losses_80842в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╞trace_0
:x
2exit_3/kernel
:
2exit_3/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╟non_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
ь
╠trace_02═
&__inference_exit_4_layer_call_fn_80851в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╠trace_0
З
═trace_02ш
A__inference_exit_4_layer_call_and_return_conditional_losses_80862в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z═trace_0
:T
2exit_4/kernel
:
2exit_4/bias
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
h
╬0
╧1
╨2
╤3
╥4
╙5
╘6
╒7
╓8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЎBє
%__inference_model_layer_call_fn_79989input"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ўBЇ
%__inference_model_layer_call_fn_80464inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ўBЇ
%__inference_model_layer_call_fn_80507inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЎBє
%__inference_model_layer_call_fn_80266input"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ТBП
@__inference_model_layer_call_and_return_conditional_losses_80578inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ТBП
@__inference_model_layer_call_and_return_conditional_losses_80649inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
СBО
@__inference_model_layer_call_and_return_conditional_losses_80318input"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
СBО
@__inference_model_layer_call_and_return_conditional_losses_80370input"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╚B┼
#__inference_signature_wrapper_80421input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╓B╙
"__inference_C1_layer_call_fn_80658inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
=__inference_C1_layer_call_and_return_conditional_losses_80669inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╓B╙
"__inference_S2_layer_call_fn_80674inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
=__inference_S2_layer_call_and_return_conditional_losses_80679inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╓B╙
"__inference_C3_layer_call_fn_80688inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
=__inference_C3_layer_call_and_return_conditional_losses_80699inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╓B╙
"__inference_S4_layer_call_fn_80704inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
=__inference_S4_layer_call_and_return_conditional_losses_80709inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_Flatten_layer_call_fn_80714inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
█B╪
'__inference_Flatten_layer_call_fn_80719inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
█B╪
'__inference_Flatten_layer_call_fn_80724inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_Flatten_layer_call_and_return_conditional_losses_80730inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_Flatten_layer_call_and_return_conditional_losses_80736inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_Flatten_layer_call_and_return_conditional_losses_80742inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╓B╙
"__inference_F5_layer_call_fn_80751inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
=__inference_F5_layer_call_and_return_conditional_losses_80762inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╓B╙
"__inference_F6_layer_call_fn_80771inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
=__inference_F6_layer_call_and_return_conditional_losses_80782inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_exit_1_layer_call_fn_80791inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_exit_1_layer_call_and_return_conditional_losses_80802inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_exit_2_layer_call_fn_80811inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_exit_2_layer_call_and_return_conditional_losses_80822inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_exit_3_layer_call_fn_80831inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_exit_3_layer_call_and_return_conditional_losses_80842inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_exit_4_layer_call_fn_80851inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_exit_4_layer_call_and_return_conditional_losses_80862inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
╫	variables
╪	keras_api

┘total

┌count"
_tf_keras_metric
R
█	variables
▄	keras_api

▌total

▐count"
_tf_keras_metric
R
▀	variables
р	keras_api

сtotal

тcount"
_tf_keras_metric
R
у	variables
ф	keras_api

хtotal

цcount"
_tf_keras_metric
R
ч	variables
ш	keras_api

щtotal

ъcount"
_tf_keras_metric
c
ы	variables
ь	keras_api

эtotal

юcount
я
_fn_kwargs"
_tf_keras_metric
c
Ё	variables
ё	keras_api

Єtotal

єcount
Ї
_fn_kwargs"
_tf_keras_metric
c
ї	variables
Ў	keras_api

ўtotal

°count
∙
_fn_kwargs"
_tf_keras_metric
c
·	variables
√	keras_api

№total

¤count
■
_fn_kwargs"
_tf_keras_metric
0
┘0
┌1"
trackable_list_wrapper
.
╫	variables"
_generic_user_object
:  (2total
:  (2count
0
▌0
▐1"
trackable_list_wrapper
.
█	variables"
_generic_user_object
:  (2total
:  (2count
0
с0
т1"
trackable_list_wrapper
.
▀	variables"
_generic_user_object
:  (2total
:  (2count
0
х0
ц1"
trackable_list_wrapper
.
у	variables"
_generic_user_object
:  (2total
:  (2count
0
щ0
ъ1"
trackable_list_wrapper
.
ч	variables"
_generic_user_object
:  (2total
:  (2count
0
э0
ю1"
trackable_list_wrapper
.
ы	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Є0
є1"
trackable_list_wrapper
.
Ё	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ў0
°1"
trackable_list_wrapper
.
ї	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
№0
¤1"
trackable_list_wrapper
.
·	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
(:&2Adam/C1/kernel/m
:2Adam/C1/bias/m
(:&2Adam/C3/kernel/m
:2Adam/C3/bias/m
!:	Аx2Adam/F5/kernel/m
:x2Adam/F5/bias/m
 :xT2Adam/F6/kernel/m
:T2Adam/F6/bias/m
%:#	А
2Adam/exit_1/kernel/m
:
2Adam/exit_1/bias/m
%:#	А
2Adam/exit_2/kernel/m
:
2Adam/exit_2/bias/m
$:"x
2Adam/exit_3/kernel/m
:
2Adam/exit_3/bias/m
$:"T
2Adam/exit_4/kernel/m
:
2Adam/exit_4/bias/m
(:&2Adam/C1/kernel/v
:2Adam/C1/bias/v
(:&2Adam/C3/kernel/v
:2Adam/C3/bias/v
!:	Аx2Adam/F5/kernel/v
:x2Adam/F5/bias/v
 :xT2Adam/F6/kernel/v
:T2Adam/F6/bias/v
%:#	А
2Adam/exit_1/kernel/v
:
2Adam/exit_1/bias/v
%:#	А
2Adam/exit_2/kernel/v
:
2Adam/exit_2/bias/v
$:"x
2Adam/exit_3/kernel/v
:
2Adam/exit_3/bias/v
$:"T
2Adam/exit_4/kernel/v
:
2Adam/exit_4/bias/vн
=__inference_C1_layer_call_and_return_conditional_losses_80669l7в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ Е
"__inference_C1_layer_call_fn_80658_7в4
-в*
(К%
inputs         
к " К         н
=__inference_C3_layer_call_and_return_conditional_losses_80699l+,7в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ Е
"__inference_C3_layer_call_fn_80688_+,7в4
-в*
(К%
inputs         
к " К         Ю
=__inference_F5_layer_call_and_return_conditional_losses_80762]@A0в-
&в#
!К
inputs         А
к "%в"
К
0         x
Ъ v
"__inference_F5_layer_call_fn_80751P@A0в-
&в#
!К
inputs         А
к "К         xЭ
=__inference_F6_layer_call_and_return_conditional_losses_80782\HI/в,
%в"
 К
inputs         x
к "%в"
К
0         T
Ъ u
"__inference_F6_layer_call_fn_80771OHI/в,
%в"
 К
inputs         x
к "К         Tз
B__inference_Flatten_layer_call_and_return_conditional_losses_80730a7в4
-в*
(К%
inputs         
к "&в#
К
0         А
Ъ з
B__inference_Flatten_layer_call_and_return_conditional_losses_80736a7в4
-в*
(К%
inputs         
к "&в#
К
0         А
Ъ з
B__inference_Flatten_layer_call_and_return_conditional_losses_80742a7в4
-в*
(К%
inputs         
к "&в#
К
0         А
Ъ 
'__inference_Flatten_layer_call_fn_80714T7в4
-в*
(К%
inputs         
к "К         А
'__inference_Flatten_layer_call_fn_80719T7в4
-в*
(К%
inputs         
к "К         А
'__inference_Flatten_layer_call_fn_80724T7в4
-в*
(К%
inputs         
к "К         Ар
=__inference_S2_layer_call_and_return_conditional_losses_80679ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╕
"__inference_S2_layer_call_fn_80674СRвO
HвE
CК@
inputs4                                    
к ";К84                                    р
=__inference_S4_layer_call_and_return_conditional_losses_80709ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╕
"__inference_S4_layer_call_fn_80704СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ж
 __inference__wrapped_model_79753Б+,@AHIhi`aXYPQ6в3
,в)
'К$
input         
к "┤к░
*
exit_1 К
exit_1         

*
exit_2 К
exit_2         

*
exit_3 К
exit_3         

*
exit_4 К
exit_4         
в
A__inference_exit_1_layer_call_and_return_conditional_losses_80802]PQ0в-
&в#
!К
inputs         А
к "%в"
К
0         

Ъ z
&__inference_exit_1_layer_call_fn_80791PPQ0в-
&в#
!К
inputs         А
к "К         
в
A__inference_exit_2_layer_call_and_return_conditional_losses_80822]XY0в-
&в#
!К
inputs         А
к "%в"
К
0         

Ъ z
&__inference_exit_2_layer_call_fn_80811PXY0в-
&в#
!К
inputs         А
к "К         
б
A__inference_exit_3_layer_call_and_return_conditional_losses_80842\`a/в,
%в"
 К
inputs         x
к "%в"
К
0         

Ъ y
&__inference_exit_3_layer_call_fn_80831O`a/в,
%в"
 К
inputs         x
к "К         
б
A__inference_exit_4_layer_call_and_return_conditional_losses_80862\hi/в,
%в"
 К
inputs         T
к "%в"
К
0         

Ъ y
&__inference_exit_4_layer_call_fn_80851Ohi/в,
%в"
 К
inputs         T
к "К         
д
@__inference_model_layer_call_and_return_conditional_losses_80318▀+,@AHIhi`aXYPQ>в;
4в1
'К$
input         
p 

 
к "КвЖ
Ъ|
К
0/0         

К
0/1         

К
0/2         

К
0/3         

Ъ д
@__inference_model_layer_call_and_return_conditional_losses_80370▀+,@AHIhi`aXYPQ>в;
4в1
'К$
input         
p

 
к "КвЖ
Ъ|
К
0/0         

К
0/1         

К
0/2         

К
0/3         

Ъ е
@__inference_model_layer_call_and_return_conditional_losses_80578р+,@AHIhi`aXYPQ?в<
5в2
(К%
inputs         
p 

 
к "КвЖ
Ъ|
К
0/0         

К
0/1         

К
0/2         

К
0/3         

Ъ е
@__inference_model_layer_call_and_return_conditional_losses_80649р+,@AHIhi`aXYPQ?в<
5в2
(К%
inputs         
p

 
к "КвЖ
Ъ|
К
0/0         

К
0/1         

К
0/2         

К
0/3         

Ъ ї
%__inference_model_layer_call_fn_79989╦+,@AHIhi`aXYPQ>в;
4в1
'К$
input         
p 

 
к "wЪt
К
0         

К
1         

К
2         

К
3         
ї
%__inference_model_layer_call_fn_80266╦+,@AHIhi`aXYPQ>в;
4в1
'К$
input         
p

 
к "wЪt
К
0         

К
1         

К
2         

К
3         
Ў
%__inference_model_layer_call_fn_80464╠+,@AHIhi`aXYPQ?в<
5в2
(К%
inputs         
p 

 
к "wЪt
К
0         

К
1         

К
2         

К
3         
Ў
%__inference_model_layer_call_fn_80507╠+,@AHIhi`aXYPQ?в<
5в2
(К%
inputs         
p

 
к "wЪt
К
0         

К
1         

К
2         

К
3         
▓
#__inference_signature_wrapper_80421К+,@AHIhi`aXYPQ?в<
в 
5к2
0
input'К$
input         "┤к░
*
exit_1 К
exit_1         

*
exit_2 К
exit_2         

*
exit_3 К
exit_3         

*
exit_4 К
exit_4         
