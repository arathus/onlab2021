Р"
Ћ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
О
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12unknown8Ј
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

: *
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
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
Ш
5token_and_position_embedding_2/embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
рд *F
shared_name75token_and_position_embedding_2/embedding_4/embeddings
С
Itoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_4/embeddings* 
_output_shapes
:
рд *
dtype0
Ц
5token_and_position_embedding_2/embedding_5/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:< *F
shared_name75token_and_position_embedding_2/embedding_5/embeddings
П
Itoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_5/embeddings*
_output_shapes

:< *
dtype0
Ю
7transformer_block_2/multi_head_attention_2/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_2/multi_head_attention_2/query/kernel
Ч
Ktransformer_block_2/multi_head_attention_2/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_2/multi_head_attention_2/query/kernel*"
_output_shapes
:  *
dtype0
Ц
5transformer_block_2/multi_head_attention_2/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_2/multi_head_attention_2/query/bias
П
Itransformer_block_2/multi_head_attention_2/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_2/multi_head_attention_2/query/bias*
_output_shapes

: *
dtype0
Ъ
5transformer_block_2/multi_head_attention_2/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_2/multi_head_attention_2/key/kernel
У
Itransformer_block_2/multi_head_attention_2/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_2/multi_head_attention_2/key/kernel*"
_output_shapes
:  *
dtype0
Т
3transformer_block_2/multi_head_attention_2/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_2/multi_head_attention_2/key/bias
Л
Gtransformer_block_2/multi_head_attention_2/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_2/multi_head_attention_2/key/bias*
_output_shapes

: *
dtype0
Ю
7transformer_block_2/multi_head_attention_2/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_2/multi_head_attention_2/value/kernel
Ч
Ktransformer_block_2/multi_head_attention_2/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_2/multi_head_attention_2/value/kernel*"
_output_shapes
:  *
dtype0
Ц
5transformer_block_2/multi_head_attention_2/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_2/multi_head_attention_2/value/bias
П
Itransformer_block_2/multi_head_attention_2/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_2/multi_head_attention_2/value/bias*
_output_shapes

: *
dtype0
ф
Btransformer_block_2/multi_head_attention_2/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_2/multi_head_attention_2/attention_output/kernel
н
Vtransformer_block_2/multi_head_attention_2/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_2/multi_head_attention_2/attention_output/kernel*"
_output_shapes
:  *
dtype0
и
@transformer_block_2/multi_head_attention_2/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_2/multi_head_attention_2/attention_output/bias
б
Ttransformer_block_2/multi_head_attention_2/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_2/multi_head_attention_2/attention_output/bias*
_output_shapes
: *
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:  *
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
: *
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:  *
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
: *
dtype0
Ж
/transformer_block_2/layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_2/layer_normalization_4/gamma
Џ
Ctransformer_block_2/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_2/layer_normalization_4/gamma*
_output_shapes
: *
dtype0
Д
.transformer_block_2/layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_2/layer_normalization_4/beta
­
Btransformer_block_2/layer_normalization_4/beta/Read/ReadVariableOpReadVariableOp.transformer_block_2/layer_normalization_4/beta*
_output_shapes
: *
dtype0
Ж
/transformer_block_2/layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_2/layer_normalization_5/gamma
Џ
Ctransformer_block_2/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_2/layer_normalization_5/gamma*
_output_shapes
: *
dtype0
Д
.transformer_block_2/layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_2/layer_normalization_5/beta
­
Btransformer_block_2/layer_normalization_5/beta/Read/ReadVariableOpReadVariableOp.transformer_block_2/layer_normalization_5/beta*
_output_shapes
: *
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

Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_10/kernel/m

*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:*
dtype0

Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/m

*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:*
dtype0
ж
<Adam/token_and_position_embedding_2/embedding_4/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
рд *M
shared_name><Adam/token_and_position_embedding_2/embedding_4/embeddings/m
Я
PAdam/token_and_position_embedding_2/embedding_4/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_4/embeddings/m* 
_output_shapes
:
рд *
dtype0
д
<Adam/token_and_position_embedding_2/embedding_5/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:< *M
shared_name><Adam/token_and_position_embedding_2/embedding_5/embeddings/m
Э
PAdam/token_and_position_embedding_2/embedding_5/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_5/embeddings/m*
_output_shapes

:< *
dtype0
м
>Adam/transformer_block_2/multi_head_attention_2/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>Adam/transformer_block_2/multi_head_attention_2/query/kernel/m
е
RAdam/transformer_block_2/multi_head_attention_2/query/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_2/multi_head_attention_2/query/kernel/m*"
_output_shapes
:  *
dtype0
д
<Adam/transformer_block_2/multi_head_attention_2/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/query/bias/m
Э
PAdam/transformer_block_2/multi_head_attention_2/query/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/query/bias/m*
_output_shapes

: *
dtype0
и
<Adam/transformer_block_2/multi_head_attention_2/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/key/kernel/m
б
PAdam/transformer_block_2/multi_head_attention_2/key/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/key/kernel/m*"
_output_shapes
:  *
dtype0
а
:Adam/transformer_block_2/multi_head_attention_2/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *K
shared_name<:Adam/transformer_block_2/multi_head_attention_2/key/bias/m
Щ
NAdam/transformer_block_2/multi_head_attention_2/key/bias/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_2/multi_head_attention_2/key/bias/m*
_output_shapes

: *
dtype0
м
>Adam/transformer_block_2/multi_head_attention_2/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>Adam/transformer_block_2/multi_head_attention_2/value/kernel/m
е
RAdam/transformer_block_2/multi_head_attention_2/value/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_2/multi_head_attention_2/value/kernel/m*"
_output_shapes
:  *
dtype0
д
<Adam/transformer_block_2/multi_head_attention_2/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/value/bias/m
Э
PAdam/transformer_block_2/multi_head_attention_2/value/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/value/bias/m*
_output_shapes

: *
dtype0
ђ
IAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *Z
shared_nameKIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/m
ы
]Adam/transformer_block_2/multi_head_attention_2/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/m*"
_output_shapes
:  *
dtype0
ц
GAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/m
п
[Adam/transformer_block_2/multi_head_attention_2/attention_output/bias/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/m*
_output_shapes
: *
dtype0

Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

:  *
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
: *
dtype0

Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:  *
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
: *
dtype0
Ф
6Adam/transformer_block_2/layer_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_4/gamma/m
Н
JAdam/transformer_block_2/layer_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_4/gamma/m*
_output_shapes
: *
dtype0
Т
5Adam/transformer_block_2/layer_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_4/beta/m
Л
IAdam/transformer_block_2/layer_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_4/beta/m*
_output_shapes
: *
dtype0
Ф
6Adam/transformer_block_2/layer_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_5/gamma/m
Н
JAdam/transformer_block_2/layer_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_5/gamma/m*
_output_shapes
: *
dtype0
Т
5Adam/transformer_block_2/layer_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_5/beta/m
Л
IAdam/transformer_block_2/layer_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_5/beta/m*
_output_shapes
: *
dtype0

Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_10/kernel/v

*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:*
dtype0

Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/v

*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:*
dtype0
ж
<Adam/token_and_position_embedding_2/embedding_4/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
рд *M
shared_name><Adam/token_and_position_embedding_2/embedding_4/embeddings/v
Я
PAdam/token_and_position_embedding_2/embedding_4/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_4/embeddings/v* 
_output_shapes
:
рд *
dtype0
д
<Adam/token_and_position_embedding_2/embedding_5/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:< *M
shared_name><Adam/token_and_position_embedding_2/embedding_5/embeddings/v
Э
PAdam/token_and_position_embedding_2/embedding_5/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_5/embeddings/v*
_output_shapes

:< *
dtype0
м
>Adam/transformer_block_2/multi_head_attention_2/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>Adam/transformer_block_2/multi_head_attention_2/query/kernel/v
е
RAdam/transformer_block_2/multi_head_attention_2/query/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_2/multi_head_attention_2/query/kernel/v*"
_output_shapes
:  *
dtype0
д
<Adam/transformer_block_2/multi_head_attention_2/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/query/bias/v
Э
PAdam/transformer_block_2/multi_head_attention_2/query/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/query/bias/v*
_output_shapes

: *
dtype0
и
<Adam/transformer_block_2/multi_head_attention_2/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/key/kernel/v
б
PAdam/transformer_block_2/multi_head_attention_2/key/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/key/kernel/v*"
_output_shapes
:  *
dtype0
а
:Adam/transformer_block_2/multi_head_attention_2/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *K
shared_name<:Adam/transformer_block_2/multi_head_attention_2/key/bias/v
Щ
NAdam/transformer_block_2/multi_head_attention_2/key/bias/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_2/multi_head_attention_2/key/bias/v*
_output_shapes

: *
dtype0
м
>Adam/transformer_block_2/multi_head_attention_2/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>Adam/transformer_block_2/multi_head_attention_2/value/kernel/v
е
RAdam/transformer_block_2/multi_head_attention_2/value/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_2/multi_head_attention_2/value/kernel/v*"
_output_shapes
:  *
dtype0
д
<Adam/transformer_block_2/multi_head_attention_2/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/value/bias/v
Э
PAdam/transformer_block_2/multi_head_attention_2/value/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/value/bias/v*
_output_shapes

: *
dtype0
ђ
IAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *Z
shared_nameKIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/v
ы
]Adam/transformer_block_2/multi_head_attention_2/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/v*"
_output_shapes
:  *
dtype0
ц
GAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/v
п
[Adam/transformer_block_2/multi_head_attention_2/attention_output/bias/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/v*
_output_shapes
: *
dtype0

Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

:  *
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
: *
dtype0

Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:  *
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
: *
dtype0
Ф
6Adam/transformer_block_2/layer_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_4/gamma/v
Н
JAdam/transformer_block_2/layer_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_4/gamma/v*
_output_shapes
: *
dtype0
Т
5Adam/transformer_block_2/layer_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_4/beta/v
Л
IAdam/transformer_block_2/layer_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_4/beta/v*
_output_shapes
: *
dtype0
Ф
6Adam/transformer_block_2/layer_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_5/gamma/v
Н
JAdam/transformer_block_2/layer_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_5/gamma/v*
_output_shapes
: *
dtype0
Т
5Adam/transformer_block_2/layer_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_5/beta/v
Л
IAdam/transformer_block_2/layer_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_5/beta/v*
_output_shapes
: *
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ю
valueУBП BЗ
С
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
n
	token_emb
pos_emb
trainable_variables
	variables
regularization_losses
	keras_api
 
att
ffn

layernorm1

layernorm2
dropout1
dropout2
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
 	variables
!regularization_losses
"	keras_api
R
#trainable_variables
$	variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
ј
7iter

8beta_1

9beta_2
	:decay
;learning_rate'mЃ(mЄ1mЅ2mІ<mЇ=mЈ>mЉ?mЊ@mЋAmЌBm­CmЎDmЏEmАFmБGmВHmГImДJmЕKmЖLmЗMmИ'vЙ(vК1vЛ2vМ<vН=vО>vП?vР@vСAvТBvУCvФDvХEvЦFvЧGvШHvЩIvЪJvЫKvЬLvЭMvЮ
І
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221
І
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221
 
­
Nlayer_metrics
Olayer_regularization_losses

trainable_variables
Pnon_trainable_variables
	variables
regularization_losses

Qlayers
Rmetrics
 
b
<
embeddings
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
b
=
embeddings
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api

<0
=1

<0
=1
 
­
[layer_metrics
\layer_regularization_losses
trainable_variables
]non_trainable_variables
	variables
regularization_losses

^layers
_metrics
Л
`_query_dense
a
_key_dense
b_value_dense
c_softmax
d_dropout_layer
e_output_dense
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
 
jlayer_with_weights-0
jlayer-0
klayer_with_weights-1
klayer-1
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
q
paxis
	Jgamma
Kbeta
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
q
uaxis
	Lgamma
Mbeta
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
R
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
T
~trainable_variables
	variables
regularization_losses
	keras_api
v
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
v
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
 
В
layer_metrics
 layer_regularization_losses
trainable_variables
non_trainable_variables
	variables
regularization_losses
layers
metrics
 
 
 
В
layer_metrics
 layer_regularization_losses
trainable_variables
non_trainable_variables
 	variables
!regularization_losses
layers
metrics
 
 
 
В
layer_metrics
 layer_regularization_losses
#trainable_variables
non_trainable_variables
$	variables
%regularization_losses
layers
metrics
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
В
layer_metrics
 layer_regularization_losses
)trainable_variables
non_trainable_variables
*	variables
+regularization_losses
layers
metrics
 
 
 
В
layer_metrics
 layer_regularization_losses
-trainable_variables
non_trainable_variables
.	variables
/regularization_losses
layers
metrics
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
В
layer_metrics
 layer_regularization_losses
3trainable_variables
non_trainable_variables
4	variables
5regularization_losses
layers
metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5token_and_position_embedding_2/embedding_4/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5token_and_position_embedding_2/embedding_5/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE7transformer_block_2/multi_head_attention_2/query/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5transformer_block_2/multi_head_attention_2/query/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5transformer_block_2/multi_head_attention_2/key/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE3transformer_block_2/multi_head_attention_2/key/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE7transformer_block_2/multi_head_attention_2/value/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5transformer_block_2/multi_head_attention_2/value/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEBtransformer_block_2/multi_head_attention_2/attention_output/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE@transformer_block_2/multi_head_attention_2/attention_output/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_8/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_8/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_9/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_9/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_2/layer_normalization_4/gamma1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.transformer_block_2/layer_normalization_4/beta1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_2/layer_normalization_5/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.transformer_block_2/layer_normalization_5/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
8
0
1
2
3
4
5
6
7

 0
Ё1

<0

<0
 
В
Ђlayer_metrics
 Ѓlayer_regularization_losses
Strainable_variables
Єnon_trainable_variables
T	variables
Uregularization_losses
Ѕlayers
Іmetrics

=0

=0
 
В
Їlayer_metrics
 Јlayer_regularization_losses
Wtrainable_variables
Љnon_trainable_variables
X	variables
Yregularization_losses
Њlayers
Ћmetrics
 
 
 

0
1
 

Ќpartial_output_shape
­full_output_shape

>kernel
?bias
Ўtrainable_variables
Џ	variables
Аregularization_losses
Б	keras_api

Вpartial_output_shape
Гfull_output_shape

@kernel
Abias
Дtrainable_variables
Е	variables
Жregularization_losses
З	keras_api

Иpartial_output_shape
Йfull_output_shape

Bkernel
Cbias
Кtrainable_variables
Л	variables
Мregularization_losses
Н	keras_api
V
Оtrainable_variables
П	variables
Рregularization_losses
С	keras_api
V
Тtrainable_variables
У	variables
Фregularization_losses
Х	keras_api

Цpartial_output_shape
Чfull_output_shape

Dkernel
Ebias
Шtrainable_variables
Щ	variables
Ъregularization_losses
Ы	keras_api
8
>0
?1
@2
A3
B4
C5
D6
E7
8
>0
?1
@2
A3
B4
C5
D6
E7
 
В
Ьlayer_metrics
 Эlayer_regularization_losses
ftrainable_variables
Юnon_trainable_variables
g	variables
hregularization_losses
Яlayers
аmetrics
l

Fkernel
Gbias
бtrainable_variables
в	variables
гregularization_losses
д	keras_api
l

Hkernel
Ibias
еtrainable_variables
ж	variables
зregularization_losses
и	keras_api

F0
G1
H2
I3

F0
G1
H2
I3
 
В
йlayer_metrics
 кlayer_regularization_losses
ltrainable_variables
лnon_trainable_variables
m	variables
nregularization_losses
мlayers
нmetrics
 

J0
K1

J0
K1
 
В
оlayer_metrics
 пlayer_regularization_losses
qtrainable_variables
рnon_trainable_variables
r	variables
sregularization_losses
сlayers
тmetrics
 

L0
M1

L0
M1
 
В
уlayer_metrics
 фlayer_regularization_losses
vtrainable_variables
хnon_trainable_variables
w	variables
xregularization_losses
цlayers
чmetrics
 
 
 
В
шlayer_metrics
 щlayer_regularization_losses
ztrainable_variables
ъnon_trainable_variables
{	variables
|regularization_losses
ыlayers
ьmetrics
 
 
 
Г
эlayer_metrics
 юlayer_regularization_losses
~trainable_variables
яnon_trainable_variables
	variables
regularization_losses
№layers
ёmetrics
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

ђtotal

ѓcount
є	variables
ѕ	keras_api
I

іtotal

їcount
ј
_fn_kwargs
љ	variables
њ	keras_api
 
 
 
 
 
 
 
 
 
 
 
 

>0
?1

>0
?1
 
Е
ћlayer_metrics
 ќlayer_regularization_losses
Ўtrainable_variables
§non_trainable_variables
Џ	variables
Аregularization_losses
ўlayers
џmetrics
 
 

@0
A1

@0
A1
 
Е
layer_metrics
 layer_regularization_losses
Дtrainable_variables
non_trainable_variables
Е	variables
Жregularization_losses
layers
metrics
 
 

B0
C1

B0
C1
 
Е
layer_metrics
 layer_regularization_losses
Кtrainable_variables
non_trainable_variables
Л	variables
Мregularization_losses
layers
metrics
 
 
 
Е
layer_metrics
 layer_regularization_losses
Оtrainable_variables
non_trainable_variables
П	variables
Рregularization_losses
layers
metrics
 
 
 
Е
layer_metrics
 layer_regularization_losses
Тtrainable_variables
non_trainable_variables
У	variables
Фregularization_losses
layers
metrics
 
 

D0
E1

D0
E1
 
Е
layer_metrics
 layer_regularization_losses
Шtrainable_variables
non_trainable_variables
Щ	variables
Ъregularization_losses
layers
metrics
 
 
 
*
`0
a1
b2
c3
d4
e5
 

F0
G1

F0
G1
 
Е
layer_metrics
 layer_regularization_losses
бtrainable_variables
non_trainable_variables
в	variables
гregularization_losses
layers
metrics

H0
I1

H0
I1
 
Е
layer_metrics
 layer_regularization_losses
еtrainable_variables
 non_trainable_variables
ж	variables
зregularization_losses
Ёlayers
Ђmetrics
 
 
 

j0
k1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ђ0
ѓ1

є	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

і0
ї1

љ	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_4/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_5/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUE>Adam/transformer_block_2/multi_head_attention_2/query/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/query/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/key/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Adam/transformer_block_2/multi_head_attention_2/key/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUE>Adam/transformer_block_2/multi_head_attention_2/value/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/value/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЌЉ
VARIABLE_VALUEIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЊЇ
VARIABLE_VALUEGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_8/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_8/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_9/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_9/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_4/gamma/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_4/beta/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_5/gamma/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_5/beta/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_4/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_5/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUE>Adam/transformer_block_2/multi_head_attention_2/query/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/query/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/key/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Adam/transformer_block_2/multi_head_attention_2/key/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ё
VARIABLE_VALUE>Adam/transformer_block_2/multi_head_attention_2/value/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/value/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЌЉ
VARIABLE_VALUEIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЊЇ
VARIABLE_VALUEGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_8/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_8/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_9/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_9/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_4/gamma/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_4/beta/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_5/gamma/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_5/beta/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_4Placeholder*'
_output_shapes
:џџџџџџџџџ<*
dtype0*
shape:џџџџџџџџџ<
к	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_45token_and_position_embedding_2/embedding_5/embeddings5token_and_position_embedding_2/embedding_4/embeddings7transformer_block_2/multi_head_attention_2/query/kernel5transformer_block_2/multi_head_attention_2/query/bias5transformer_block_2/multi_head_attention_2/key/kernel3transformer_block_2/multi_head_attention_2/key/bias7transformer_block_2/multi_head_attention_2/value/kernel5transformer_block_2/multi_head_attention_2/value/biasBtransformer_block_2/multi_head_attention_2/attention_output/kernel@transformer_block_2/multi_head_attention_2/attention_output/bias/transformer_block_2/layer_normalization_4/gamma.transformer_block_2/layer_normalization_4/betadense_8/kerneldense_8/biasdense_9/kerneldense_9/bias/transformer_block_2/layer_normalization_5/gamma.transformer_block_2/layer_normalization_5/betadense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_9253
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
д&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpKtransformer_block_2/multi_head_attention_2/query/kernel/Read/ReadVariableOpItransformer_block_2/multi_head_attention_2/query/bias/Read/ReadVariableOpItransformer_block_2/multi_head_attention_2/key/kernel/Read/ReadVariableOpGtransformer_block_2/multi_head_attention_2/key/bias/Read/ReadVariableOpKtransformer_block_2/multi_head_attention_2/value/kernel/Read/ReadVariableOpItransformer_block_2/multi_head_attention_2/value/bias/Read/ReadVariableOpVtransformer_block_2/multi_head_attention_2/attention_output/kernel/Read/ReadVariableOpTtransformer_block_2/multi_head_attention_2/attention_output/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpCtransformer_block_2/layer_normalization_4/gamma/Read/ReadVariableOpBtransformer_block_2/layer_normalization_4/beta/Read/ReadVariableOpCtransformer_block_2/layer_normalization_5/gamma/Read/ReadVariableOpBtransformer_block_2/layer_normalization_5/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_4/embeddings/m/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_5/embeddings/m/Read/ReadVariableOpRAdam/transformer_block_2/multi_head_attention_2/query/kernel/m/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/query/bias/m/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/key/kernel/m/Read/ReadVariableOpNAdam/transformer_block_2/multi_head_attention_2/key/bias/m/Read/ReadVariableOpRAdam/transformer_block_2/multi_head_attention_2/value/kernel/m/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/value/bias/m/Read/ReadVariableOp]Adam/transformer_block_2/multi_head_attention_2/attention_output/kernel/m/Read/ReadVariableOp[Adam/transformer_block_2/multi_head_attention_2/attention_output/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_4/gamma/m/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_4/beta/m/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_5/gamma/m/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_5/beta/m/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_4/embeddings/v/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_5/embeddings/v/Read/ReadVariableOpRAdam/transformer_block_2/multi_head_attention_2/query/kernel/v/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/query/bias/v/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/key/kernel/v/Read/ReadVariableOpNAdam/transformer_block_2/multi_head_attention_2/key/bias/v/Read/ReadVariableOpRAdam/transformer_block_2/multi_head_attention_2/value/kernel/v/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/value/bias/v/Read/ReadVariableOp]Adam/transformer_block_2/multi_head_attention_2/attention_output/kernel/v/Read/ReadVariableOp[Adam/transformer_block_2/multi_head_attention_2/attention_output/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_4/gamma/v/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_4/beta/v/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_5/gamma/v/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_5/beta/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_10681
ѓ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10/kerneldense_10/biasdense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate5token_and_position_embedding_2/embedding_4/embeddings5token_and_position_embedding_2/embedding_5/embeddings7transformer_block_2/multi_head_attention_2/query/kernel5transformer_block_2/multi_head_attention_2/query/bias5transformer_block_2/multi_head_attention_2/key/kernel3transformer_block_2/multi_head_attention_2/key/bias7transformer_block_2/multi_head_attention_2/value/kernel5transformer_block_2/multi_head_attention_2/value/biasBtransformer_block_2/multi_head_attention_2/attention_output/kernel@transformer_block_2/multi_head_attention_2/attention_output/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias/transformer_block_2/layer_normalization_4/gamma.transformer_block_2/layer_normalization_4/beta/transformer_block_2/layer_normalization_5/gamma.transformer_block_2/layer_normalization_5/betatotalcounttotal_1count_1Adam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/m<Adam/token_and_position_embedding_2/embedding_4/embeddings/m<Adam/token_and_position_embedding_2/embedding_5/embeddings/m>Adam/transformer_block_2/multi_head_attention_2/query/kernel/m<Adam/transformer_block_2/multi_head_attention_2/query/bias/m<Adam/transformer_block_2/multi_head_attention_2/key/kernel/m:Adam/transformer_block_2/multi_head_attention_2/key/bias/m>Adam/transformer_block_2/multi_head_attention_2/value/kernel/m<Adam/transformer_block_2/multi_head_attention_2/value/bias/mIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/mGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/m6Adam/transformer_block_2/layer_normalization_4/gamma/m5Adam/transformer_block_2/layer_normalization_4/beta/m6Adam/transformer_block_2/layer_normalization_5/gamma/m5Adam/transformer_block_2/layer_normalization_5/beta/mAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/v<Adam/token_and_position_embedding_2/embedding_4/embeddings/v<Adam/token_and_position_embedding_2/embedding_5/embeddings/v>Adam/transformer_block_2/multi_head_attention_2/query/kernel/v<Adam/transformer_block_2/multi_head_attention_2/query/bias/v<Adam/transformer_block_2/multi_head_attention_2/key/kernel/v:Adam/transformer_block_2/multi_head_attention_2/key/bias/v>Adam/transformer_block_2/multi_head_attention_2/value/kernel/v<Adam/transformer_block_2/multi_head_attention_2/value/bias/vIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/vGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v6Adam/transformer_block_2/layer_normalization_4/gamma/v5Adam/transformer_block_2/layer_normalization_4/beta/v6Adam/transformer_block_2/layer_normalization_5/gamma/v5Adam/transformer_block_2/layer_normalization_5/beta/v*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_10916§
Ъ

п
3__inference_transformer_block_2_layer_call_fn_10098

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identityЂStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_86942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:џџџџџџџџџ< ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
МЈ
+
__inference__traced_save_10681
file_prefix.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopT
Psavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopT
Psavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableopV
Rsavev2_transformer_block_2_multi_head_attention_2_query_kernel_read_readvariableopT
Psavev2_transformer_block_2_multi_head_attention_2_query_bias_read_readvariableopT
Psavev2_transformer_block_2_multi_head_attention_2_key_kernel_read_readvariableopR
Nsavev2_transformer_block_2_multi_head_attention_2_key_bias_read_readvariableopV
Rsavev2_transformer_block_2_multi_head_attention_2_value_kernel_read_readvariableopT
Psavev2_transformer_block_2_multi_head_attention_2_value_bias_read_readvariableopa
]savev2_transformer_block_2_multi_head_attention_2_attention_output_kernel_read_readvariableop_
[savev2_transformer_block_2_multi_head_attention_2_attention_output_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableopN
Jsavev2_transformer_block_2_layer_normalization_4_gamma_read_readvariableopM
Isavev2_transformer_block_2_layer_normalization_4_beta_read_readvariableopN
Jsavev2_transformer_block_2_layer_normalization_5_gamma_read_readvariableopM
Isavev2_transformer_block_2_layer_normalization_5_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_m_read_readvariableop]
Ysavev2_adam_transformer_block_2_multi_head_attention_2_query_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_query_bias_m_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_key_kernel_m_read_readvariableopY
Usavev2_adam_transformer_block_2_multi_head_attention_2_key_bias_m_read_readvariableop]
Ysavev2_adam_transformer_block_2_multi_head_attention_2_value_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_value_bias_m_read_readvariableoph
dsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_m_read_readvariableopf
bsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_4_gamma_m_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_4_beta_m_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_5_gamma_m_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_5_beta_m_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_v_read_readvariableop]
Ysavev2_adam_transformer_block_2_multi_head_attention_2_query_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_query_bias_v_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_key_kernel_v_read_readvariableopY
Usavev2_adam_transformer_block_2_multi_head_attention_2_key_bias_v_read_readvariableop]
Ysavev2_adam_transformer_block_2_multi_head_attention_2_value_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_value_bias_v_read_readvariableoph
dsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_v_read_readvariableopf
bsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_4_gamma_v_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_4_beta_v_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_5_gamma_v_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_5_beta_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameц(
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*ј'
valueю'Bы'LB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЃ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
valueЃB LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices*
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableopRsavev2_transformer_block_2_multi_head_attention_2_query_kernel_read_readvariableopPsavev2_transformer_block_2_multi_head_attention_2_query_bias_read_readvariableopPsavev2_transformer_block_2_multi_head_attention_2_key_kernel_read_readvariableopNsavev2_transformer_block_2_multi_head_attention_2_key_bias_read_readvariableopRsavev2_transformer_block_2_multi_head_attention_2_value_kernel_read_readvariableopPsavev2_transformer_block_2_multi_head_attention_2_value_bias_read_readvariableop]savev2_transformer_block_2_multi_head_attention_2_attention_output_kernel_read_readvariableop[savev2_transformer_block_2_multi_head_attention_2_attention_output_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableopJsavev2_transformer_block_2_layer_normalization_4_gamma_read_readvariableopIsavev2_transformer_block_2_layer_normalization_4_beta_read_readvariableopJsavev2_transformer_block_2_layer_normalization_5_gamma_read_readvariableopIsavev2_transformer_block_2_layer_normalization_5_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_m_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_m_read_readvariableopYsavev2_adam_transformer_block_2_multi_head_attention_2_query_kernel_m_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_query_bias_m_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_key_kernel_m_read_readvariableopUsavev2_adam_transformer_block_2_multi_head_attention_2_key_bias_m_read_readvariableopYsavev2_adam_transformer_block_2_multi_head_attention_2_value_kernel_m_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_value_bias_m_read_readvariableopdsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_m_read_readvariableopbsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_4_gamma_m_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_4_beta_m_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_5_gamma_m_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_5_beta_m_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_v_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_v_read_readvariableopYsavev2_adam_transformer_block_2_multi_head_attention_2_query_kernel_v_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_query_bias_v_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_key_kernel_v_read_readvariableopUsavev2_adam_transformer_block_2_multi_head_attention_2_key_bias_v_read_readvariableopYsavev2_adam_transformer_block_2_multi_head_attention_2_value_kernel_v_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_value_bias_v_read_readvariableopdsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_v_read_readvariableopbsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_4_gamma_v_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_4_beta_v_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_5_gamma_v_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_5_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesї
є: : :::: : : : : :
рд :< :  : :  : :  : :  : :  : :  : : : : : : : : : : ::::
рд :< :  : :  : :  : :  : :  : :  : : : : : : ::::
рд :< :  : :  : :  : :  : :  : :  : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :&
"
 
_output_shapes
:
рд :$ 

_output_shapes

:< :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :
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
: :$  

_output_shapes

: : !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::&$"
 
_output_shapes
:
рд :$% 

_output_shapes

:< :(&$
"
_output_shapes
:  :$' 

_output_shapes

: :(($
"
_output_shapes
:  :$) 

_output_shapes

: :(*$
"
_output_shapes
:  :$+ 

_output_shapes

: :(,$
"
_output_shapes
:  : -

_output_shapes
: :$. 

_output_shapes

:  : /

_output_shapes
: :$0 

_output_shapes

:  : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: :$6 

_output_shapes

: : 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
::&:"
 
_output_shapes
:
рд :$; 

_output_shapes

:< :(<$
"
_output_shapes
:  :$= 

_output_shapes

: :(>$
"
_output_shapes
:  :$? 

_output_shapes

: :(@$
"
_output_shapes
:  :$A 

_output_shapes

: :(B$
"
_output_shapes
:  : C

_output_shapes
: :$D 

_output_shapes

:  : E

_output_shapes
: :$F 

_output_shapes

:  : G

_output_shapes
: : H

_output_shapes
: : I

_output_shapes
: : J

_output_shapes
: : K

_output_shapes
: :L

_output_shapes
: 
­ќ
а
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_9897

inputsF
Bmulti_head_attention_2_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_query_add_readvariableop_resourceD
@multi_head_attention_2_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_2_key_add_readvariableop_resourceF
Bmulti_head_attention_2_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_value_add_readvariableop_resourceQ
Mmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_2_attention_output_add_readvariableop_resource?
;layer_normalization_4_batchnorm_mul_readvariableop_resource;
7layer_normalization_4_batchnorm_readvariableop_resource:
6sequential_2_dense_8_tensordot_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resource:
6sequential_2_dense_9_tensordot_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resource?
;layer_normalization_5_batchnorm_mul_readvariableop_resource;
7layer_normalization_5_batchnorm_readvariableop_resource
identityЂ.layer_normalization_4/batchnorm/ReadVariableOpЂ2layer_normalization_4/batchnorm/mul/ReadVariableOpЂ.layer_normalization_5/batchnorm/ReadVariableOpЂ2layer_normalization_5/batchnorm/mul/ReadVariableOpЂ:multi_head_attention_2/attention_output/add/ReadVariableOpЂDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpЂ-multi_head_attention_2/key/add/ReadVariableOpЂ7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpЂ/multi_head_attention_2/query/add/ReadVariableOpЂ9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpЂ/multi_head_attention_2/value/add/ReadVariableOpЂ9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpЂ+sequential_2/dense_8/BiasAdd/ReadVariableOpЂ-sequential_2/dense_8/Tensordot/ReadVariableOpЂ+sequential_2/dense_9/BiasAdd/ReadVariableOpЂ-sequential_2/dense_9/Tensordot/ReadVariableOp§
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_2/query/einsum/EinsumEinsuminputsAmulti_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2,
*multi_head_attention_2/query/einsum/Einsumл
/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp8multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_2/query/add/ReadVariableOpѕ
 multi_head_attention_2/query/addAddV23multi_head_attention_2/query/einsum/Einsum:output:07multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2"
 multi_head_attention_2/query/addї
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_2/key/einsum/EinsumEinsuminputs?multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2*
(multi_head_attention_2/key/einsum/Einsumе
-multi_head_attention_2/key/add/ReadVariableOpReadVariableOp6multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_2/key/add/ReadVariableOpэ
multi_head_attention_2/key/addAddV21multi_head_attention_2/key/einsum/Einsum:output:05multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2 
multi_head_attention_2/key/add§
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_2/value/einsum/EinsumEinsuminputsAmulti_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2,
*multi_head_attention_2/value/einsum/Einsumл
/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp8multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_2/value/add/ReadVariableOpѕ
 multi_head_attention_2/value/addAddV23multi_head_attention_2/value/einsum/Einsum:output:07multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2"
 multi_head_attention_2/value/add
multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓ5>2
multi_head_attention_2/Mul/yЦ
multi_head_attention_2/MulMul$multi_head_attention_2/query/add:z:0%multi_head_attention_2/Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2
multi_head_attention_2/Mulќ
$multi_head_attention_2/einsum/EinsumEinsum"multi_head_attention_2/key/add:z:0multi_head_attention_2/Mul:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ<<*
equationaecd,abcd->acbe2&
$multi_head_attention_2/einsum/EinsumФ
&multi_head_attention_2/softmax/SoftmaxSoftmax-multi_head_attention_2/einsum/Einsum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2(
&multi_head_attention_2/softmax/SoftmaxЁ
,multi_head_attention_2/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_2/dropout/dropout/Const
*multi_head_attention_2/dropout/dropout/MulMul0multi_head_attention_2/softmax/Softmax:softmax:05multi_head_attention_2/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2,
*multi_head_attention_2/dropout/dropout/MulМ
,multi_head_attention_2/dropout/dropout/ShapeShape0multi_head_attention_2/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_2/dropout/dropout/Shape
Cmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_2/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<*
dtype02E
Cmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniformГ
5multi_head_attention_2/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_2/dropout/dropout/GreaterEqual/yТ
3multi_head_attention_2/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_2/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<25
3multi_head_attention_2/dropout/dropout/GreaterEqualф
+multi_head_attention_2/dropout/dropout/CastCast7multi_head_attention_2/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ<<2-
+multi_head_attention_2/dropout/dropout/Castў
,multi_head_attention_2/dropout/dropout/Mul_1Mul.multi_head_attention_2/dropout/dropout/Mul:z:0/multi_head_attention_2/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2.
,multi_head_attention_2/dropout/dropout/Mul_1
&multi_head_attention_2/einsum_1/EinsumEinsum0multi_head_attention_2/dropout/dropout/Mul_1:z:0$multi_head_attention_2/value/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationacbe,aecd->abcd2(
&multi_head_attention_2/einsum_1/Einsum
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpг
5multi_head_attention_2/attention_output/einsum/EinsumEinsum/multi_head_attention_2/einsum_1/Einsum:output:0Lmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ< *
equationabcd,cde->abe27
5multi_head_attention_2/attention_output/einsum/Einsumј
:multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_2/attention_output/add/ReadVariableOp
+multi_head_attention_2/attention_output/addAddV2>multi_head_attention_2/attention_output/einsum/Einsum:output:0Bmulti_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2-
+multi_head_attention_2/attention_output/addw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout_8/dropout/ConstО
dropout_8/dropout/MulMul/multi_head_attention_2/attention_output/add:z:0 dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_8/dropout/Mul
dropout_8/dropout/ShapeShape/multi_head_attention_2/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shapeж
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< *
dtype020
.dropout_8/dropout/random_uniform/RandomUniform
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2"
 dropout_8/dropout/GreaterEqual/yъ
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
dropout_8/dropout/GreaterEqualЁ
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ< 2
dropout_8/dropout/CastІ
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_8/dropout/Mul_1n
addAddV2inputsdropout_8/dropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
addЖ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesп
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2$
"layer_normalization_4/moments/meanЫ
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2,
*layer_normalization_4/moments/StopGradientы
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 21
/layer_normalization_4/moments/SquaredDifferenceО
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indices
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2(
&layer_normalization_4/moments/variance
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752'
%layer_normalization_4/batchnorm/add/yъ
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2%
#layer_normalization_4/batchnorm/addЖ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2'
%layer_normalization_4/batchnorm/Rsqrtр
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpю
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_4/batchnorm/mulН
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/mul_1с
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/mul_2д
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpъ
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_4/batchnorm/subс
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/add_1е
-sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_2/dense_8/Tensordot/ReadVariableOp
#sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_8/Tensordot/axes
#sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_8/Tensordot/freeЅ
$sequential_2/dense_8/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/Shape
,sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/GatherV2/axisК
'sequential_2/dense_8/Tensordot/GatherV2GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/free:output:05sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/GatherV2Ђ
.sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_8/Tensordot/GatherV2_1/axisР
)sequential_2/dense_8/Tensordot/GatherV2_1GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/axes:output:07sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_8/Tensordot/GatherV2_1
$sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_8/Tensordot/Constд
#sequential_2/dense_8/Tensordot/ProdProd0sequential_2/dense_8/Tensordot/GatherV2:output:0-sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_8/Tensordot/Prod
&sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_1м
%sequential_2/dense_8/Tensordot/Prod_1Prod2sequential_2/dense_8/Tensordot/GatherV2_1:output:0/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_8/Tensordot/Prod_1
*sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_8/Tensordot/concat/axis
%sequential_2/dense_8/Tensordot/concatConcatV2,sequential_2/dense_8/Tensordot/free:output:0,sequential_2/dense_8/Tensordot/axes:output:03sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_8/Tensordot/concatр
$sequential_2/dense_8/Tensordot/stackPack,sequential_2/dense_8/Tensordot/Prod:output:0.sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/stackђ
(sequential_2/dense_8/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0.sequential_2/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2*
(sequential_2/dense_8/Tensordot/transposeѓ
&sequential_2/dense_8/Tensordot/ReshapeReshape,sequential_2/dense_8/Tensordot/transpose:y:0-sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2(
&sequential_2/dense_8/Tensordot/Reshapeђ
%sequential_2/dense_8/Tensordot/MatMulMatMul/sequential_2/dense_8/Tensordot/Reshape:output:05sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_2/dense_8/Tensordot/MatMul
&sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_2
,sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/concat_1/axisІ
'sequential_2/dense_8/Tensordot/concat_1ConcatV20sequential_2/dense_8/Tensordot/GatherV2:output:0/sequential_2/dense_8/Tensordot/Const_2:output:05sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/concat_1ф
sequential_2/dense_8/TensordotReshape/sequential_2/dense_8/Tensordot/MatMul:product:00sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
sequential_2/dense_8/TensordotЫ
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOpл
sequential_2/dense_8/BiasAddBiasAdd'sequential_2/dense_8/Tensordot:output:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_8/BiasAdd
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_8/Reluе
-sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_2/dense_9/Tensordot/ReadVariableOp
#sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_9/Tensordot/axes
#sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_9/Tensordot/freeЃ
$sequential_2/dense_9/Tensordot/ShapeShape'sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/Shape
,sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/GatherV2/axisК
'sequential_2/dense_9/Tensordot/GatherV2GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/free:output:05sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/GatherV2Ђ
.sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_9/Tensordot/GatherV2_1/axisР
)sequential_2/dense_9/Tensordot/GatherV2_1GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/axes:output:07sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_9/Tensordot/GatherV2_1
$sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_9/Tensordot/Constд
#sequential_2/dense_9/Tensordot/ProdProd0sequential_2/dense_9/Tensordot/GatherV2:output:0-sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_9/Tensordot/Prod
&sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_1м
%sequential_2/dense_9/Tensordot/Prod_1Prod2sequential_2/dense_9/Tensordot/GatherV2_1:output:0/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_9/Tensordot/Prod_1
*sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_9/Tensordot/concat/axis
%sequential_2/dense_9/Tensordot/concatConcatV2,sequential_2/dense_9/Tensordot/free:output:0,sequential_2/dense_9/Tensordot/axes:output:03sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_9/Tensordot/concatр
$sequential_2/dense_9/Tensordot/stackPack,sequential_2/dense_9/Tensordot/Prod:output:0.sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/stack№
(sequential_2/dense_9/Tensordot/transpose	Transpose'sequential_2/dense_8/Relu:activations:0.sequential_2/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2*
(sequential_2/dense_9/Tensordot/transposeѓ
&sequential_2/dense_9/Tensordot/ReshapeReshape,sequential_2/dense_9/Tensordot/transpose:y:0-sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2(
&sequential_2/dense_9/Tensordot/Reshapeђ
%sequential_2/dense_9/Tensordot/MatMulMatMul/sequential_2/dense_9/Tensordot/Reshape:output:05sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_2/dense_9/Tensordot/MatMul
&sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_2
,sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/concat_1/axisІ
'sequential_2/dense_9/Tensordot/concat_1ConcatV20sequential_2/dense_9/Tensordot/GatherV2:output:0/sequential_2/dense_9/Tensordot/Const_2:output:05sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/concat_1ф
sequential_2/dense_9/TensordotReshape/sequential_2/dense_9/Tensordot/MatMul:product:00sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
sequential_2/dense_9/TensordotЫ
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOpл
sequential_2/dense_9/BiasAddBiasAdd'sequential_2/dense_9/Tensordot:output:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_9/BiasAddw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout_9/dropout/ConstД
dropout_9/dropout/MulMul%sequential_2/dense_9/BiasAdd:output:0 dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_9/dropout/Mul
dropout_9/dropout/ShapeShape%sequential_2/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shapeж
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< *
dtype020
.dropout_9/dropout/random_uniform/RandomUniform
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2"
 dropout_9/dropout/GreaterEqual/yъ
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
dropout_9/dropout/GreaterEqualЁ
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ< 2
dropout_9/dropout/CastІ
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_9/dropout/Mul_1
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
add_1Ж
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indicesс
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2$
"layer_normalization_5/moments/meanЫ
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2,
*layer_normalization_5/moments/StopGradientэ
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 21
/layer_normalization_5/moments/SquaredDifferenceО
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indices
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2(
&layer_normalization_5/moments/variance
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752'
%layer_normalization_5/batchnorm/add/yъ
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2%
#layer_normalization_5/batchnorm/addЖ
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2'
%layer_normalization_5/batchnorm/Rsqrtр
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOpю
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_5/batchnorm/mulП
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/mul_1с
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/mul_2д
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOpъ
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_5/batchnorm/subс
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/add_1г
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp;^multi_head_attention_2/attention_output/add/ReadVariableOpE^multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_2/key/add/ReadVariableOp8^multi_head_attention_2/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/query/add/ReadVariableOp:^multi_head_attention_2/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/value/add/ReadVariableOp:^multi_head_attention_2/value/einsum/Einsum/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp.^sequential_2/dense_8/Tensordot/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp.^sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:џџџџџџџџџ< ::::::::::::::::2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_2/attention_output/add/ReadVariableOp:multi_head_attention_2/attention_output/add/ReadVariableOp2
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_2/key/add/ReadVariableOp-multi_head_attention_2/key/add/ReadVariableOp2r
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/query/add/ReadVariableOp/multi_head_attention_2/query/add/ReadVariableOp2v
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/value/add/ReadVariableOp/multi_head_attention_2/value/add/ReadVariableOp2v
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2^
-sequential_2/dense_8/Tensordot/ReadVariableOp-sequential_2/dense_8/Tensordot/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2^
-sequential_2/dense_9/Tensordot/ReadVariableOp-sequential_2/dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
Е
ё
F__inference_sequential_2_layer_call_and_return_conditional_losses_8345

inputs
dense_8_8334
dense_8_8336
dense_9_8339
dense_9_8341
identityЂdense_8/StatefulPartitionedCallЂdense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_8334dense_8_8336*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_82242!
dense_8/StatefulPartitionedCallЏ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_8339dense_9_8341*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_82702!
dense_9/StatefulPartitionedCallФ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ< ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
Ъ

п
3__inference_transformer_block_2_layer_call_fn_10061

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identityЂStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_85672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:џџџџџџџџџ< ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
Х

A__inference_model_2_layer_call_and_return_conditional_losses_9453

inputsD
@token_and_position_embedding_2_embedding_5_embedding_lookup_9264D
@token_and_position_embedding_2_embedding_4_embedding_lookup_9270Z
Vtransformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_2_multi_head_attention_2_query_add_readvariableop_resourceX
Ttransformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_2_multi_head_attention_2_key_add_readvariableop_resourceZ
Vtransformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_2_multi_head_attention_2_value_add_readvariableop_resourcee
atransformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resourceS
Otransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resourceN
Jtransformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resourceL
Htransformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resourceN
Jtransformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resourceL
Htransformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resourceS
Otransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identityЂdense_10/BiasAdd/ReadVariableOpЂdense_10/MatMul/ReadVariableOpЂdense_11/BiasAdd/ReadVariableOpЂdense_11/MatMul/ReadVariableOpЂ;token_and_position_embedding_2/embedding_4/embedding_lookupЂ;token_and_position_embedding_2/embedding_5/embedding_lookupЂBtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpЂFtransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpЂBtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpЂFtransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpЂNtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpЂXtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpЂAtransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpЂKtransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpЂCtransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpЂMtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpЂCtransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpЂMtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpЂ?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpЂAtransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpЂ?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpЂAtransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp
$token_and_position_embedding_2/ShapeShapeinputs*
T0*
_output_shapes
:2&
$token_and_position_embedding_2/ShapeЛ
2token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ24
2token_and_position_embedding_2/strided_slice/stackЖ
4token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_2/strided_slice/stack_1Ж
4token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_2/strided_slice/stack_2
,token_and_position_embedding_2/strided_sliceStridedSlice-token_and_position_embedding_2/Shape:output:0;token_and_position_embedding_2/strided_slice/stack:output:0=token_and_position_embedding_2/strided_slice/stack_1:output:0=token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_2/strided_slice
*token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_2/range/start
*token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_2/range/delta
$token_and_position_embedding_2/rangeRange3token_and_position_embedding_2/range/start:output:05token_and_position_embedding_2/strided_slice:output:03token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:џџџџџџџџџ2&
$token_and_position_embedding_2/rangeЦ
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGather@token_and_position_embedding_2_embedding_5_embedding_lookup_9264-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*S
_classI
GEloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/9264*'
_output_shapes
:џџџџџџџџџ *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookup
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*S
_classI
GEloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/9264*'
_output_shapes
:џџџџџџџџџ 2F
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2H
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1Г
/token_and_position_embedding_2/embedding_4/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ<21
/token_and_position_embedding_2/embedding_4/Castа
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGather@token_and_position_embedding_2_embedding_4_embedding_lookup_92703token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*S
_classI
GEloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/9270*+
_output_shapes
:џџџџџџџџџ< *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookup
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*S
_classI
GEloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/9270*+
_output_shapes
:џџџџџџџџџ< 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityЁ
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1Љ
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2$
"token_and_position_embedding_2/addЙ
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpщ
>transformer_block_2/multi_head_attention_2/query/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Utransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2@
>transformer_block_2/multi_head_attention_2/query/einsum/Einsum
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpReadVariableOpLtransformer_block_2_multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpХ
4transformer_block_2/multi_head_attention_2/query/addAddV2Gtransformer_block_2/multi_head_attention_2/query/einsum/Einsum:output:0Ktransformer_block_2/multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 26
4transformer_block_2/multi_head_attention_2/query/addГ
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpу
<transformer_block_2/multi_head_attention_2/key/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Stransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2>
<transformer_block_2/multi_head_attention_2/key/einsum/Einsum
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpReadVariableOpJtransformer_block_2_multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpН
2transformer_block_2/multi_head_attention_2/key/addAddV2Etransformer_block_2/multi_head_attention_2/key/einsum/Einsum:output:0Itransformer_block_2/multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 24
2transformer_block_2/multi_head_attention_2/key/addЙ
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpщ
>transformer_block_2/multi_head_attention_2/value/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Utransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2@
>transformer_block_2/multi_head_attention_2/value/einsum/Einsum
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpReadVariableOpLtransformer_block_2_multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpХ
4transformer_block_2/multi_head_attention_2/value/addAddV2Gtransformer_block_2/multi_head_attention_2/value/einsum/Einsum:output:0Ktransformer_block_2/multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 26
4transformer_block_2/multi_head_attention_2/value/addЉ
0transformer_block_2/multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓ5>22
0transformer_block_2/multi_head_attention_2/Mul/y
.transformer_block_2/multi_head_attention_2/MulMul8transformer_block_2/multi_head_attention_2/query/add:z:09transformer_block_2/multi_head_attention_2/Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ< 20
.transformer_block_2/multi_head_attention_2/MulЬ
8transformer_block_2/multi_head_attention_2/einsum/EinsumEinsum6transformer_block_2/multi_head_attention_2/key/add:z:02transformer_block_2/multi_head_attention_2/Mul:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ<<*
equationaecd,abcd->acbe2:
8transformer_block_2/multi_head_attention_2/einsum/Einsum
:transformer_block_2/multi_head_attention_2/softmax/SoftmaxSoftmaxAtransformer_block_2/multi_head_attention_2/einsum/Einsum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2<
:transformer_block_2/multi_head_attention_2/softmax/SoftmaxЩ
@transformer_block_2/multi_head_attention_2/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@transformer_block_2/multi_head_attention_2/dropout/dropout/Constв
>transformer_block_2/multi_head_attention_2/dropout/dropout/MulMulDtransformer_block_2/multi_head_attention_2/softmax/Softmax:softmax:0Itransformer_block_2/multi_head_attention_2/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2@
>transformer_block_2/multi_head_attention_2/dropout/dropout/Mulј
@transformer_block_2/multi_head_attention_2/dropout/dropout/ShapeShapeDtransformer_block_2/multi_head_attention_2/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_2/multi_head_attention_2/dropout/dropout/Shapeе
Wtransformer_block_2/multi_head_attention_2/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_2/multi_head_attention_2/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<*
dtype02Y
Wtransformer_block_2/multi_head_attention_2/dropout/dropout/random_uniform/RandomUniformл
Itransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqual/y
Gtransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_2/multi_head_attention_2/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2I
Gtransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqual 
?transformer_block_2/multi_head_attention_2/dropout/dropout/CastCastKtransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ<<2A
?transformer_block_2/multi_head_attention_2/dropout/dropout/CastЮ
@transformer_block_2/multi_head_attention_2/dropout/dropout/Mul_1MulBtransformer_block_2/multi_head_attention_2/dropout/dropout/Mul:z:0Ctransformer_block_2/multi_head_attention_2/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2B
@transformer_block_2/multi_head_attention_2/dropout/dropout/Mul_1ф
:transformer_block_2/multi_head_attention_2/einsum_1/EinsumEinsumDtransformer_block_2/multi_head_attention_2/dropout/dropout/Mul_1:z:08transformer_block_2/multi_head_attention_2/value/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationacbe,aecd->abcd2<
:transformer_block_2/multi_head_attention_2/einsum_1/Einsumк
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpЃ
Itransformer_block_2/multi_head_attention_2/attention_output/einsum/EinsumEinsumCtransformer_block_2/multi_head_attention_2/einsum_1/Einsum:output:0`transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ< *
equationabcd,cde->abe2K
Itransformer_block_2/multi_head_attention_2/attention_output/einsum/EinsumД
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpэ
?transformer_block_2/multi_head_attention_2/attention_output/addAddV2Rtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum:output:0Vtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2A
?transformer_block_2/multi_head_attention_2/attention_output/add
+transformer_block_2/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2-
+transformer_block_2/dropout_8/dropout/Const
)transformer_block_2/dropout_8/dropout/MulMulCtransformer_block_2/multi_head_attention_2/attention_output/add:z:04transformer_block_2/dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2+
)transformer_block_2/dropout_8/dropout/MulЭ
+transformer_block_2/dropout_8/dropout/ShapeShapeCtransformer_block_2/multi_head_attention_2/attention_output/add:z:0*
T0*
_output_shapes
:2-
+transformer_block_2/dropout_8/dropout/Shape
Btransformer_block_2/dropout_8/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_2/dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< *
dtype02D
Btransformer_block_2/dropout_8/dropout/random_uniform/RandomUniformБ
4transformer_block_2/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=26
4transformer_block_2/dropout_8/dropout/GreaterEqual/yК
2transformer_block_2/dropout_8/dropout/GreaterEqualGreaterEqualKtransformer_block_2/dropout_8/dropout/random_uniform/RandomUniform:output:0=transformer_block_2/dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 24
2transformer_block_2/dropout_8/dropout/GreaterEqualн
*transformer_block_2/dropout_8/dropout/CastCast6transformer_block_2/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ< 2,
*transformer_block_2/dropout_8/dropout/Castі
+transformer_block_2/dropout_8/dropout/Mul_1Mul-transformer_block_2/dropout_8/dropout/Mul:z:0.transformer_block_2/dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2-
+transformer_block_2/dropout_8/dropout/Mul_1Ъ
transformer_block_2/addAddV2&token_and_position_embedding_2/add:z:0/transformer_block_2/dropout_8/dropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
transformer_block_2/addо
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indicesЏ
6transformer_block_2/layer_normalization_4/moments/meanMeantransformer_block_2/add:z:0Qtransformer_block_2/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(28
6transformer_block_2/layer_normalization_4/moments/mean
>transformer_block_2/layer_normalization_4/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2@
>transformer_block_2/layer_normalization_4/moments/StopGradientЛ
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add:z:0Gtransformer_block_2/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2E
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifferenceц
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indicesч
:transformer_block_2/layer_normalization_4/moments/varianceMeanGtransformer_block_2/layer_normalization_4/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2<
:transformer_block_2/layer_normalization_4/moments/varianceЛ
9transformer_block_2/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752;
9transformer_block_2/layer_normalization_4/batchnorm/add/yК
7transformer_block_2/layer_normalization_4/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_4/moments/variance:output:0Btransformer_block_2/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<29
7transformer_block_2/layer_normalization_4/batchnorm/addђ
9transformer_block_2/layer_normalization_4/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2;
9transformer_block_2/layer_normalization_4/batchnorm/Rsqrt
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpО
7transformer_block_2/layer_normalization_4/batchnorm/mulMul=transformer_block_2/layer_normalization_4/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 29
7transformer_block_2/layer_normalization_4/batchnorm/mul
9transformer_block_2/layer_normalization_4/batchnorm/mul_1Multransformer_block_2/add:z:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_1Б
9transformer_block_2/layer_normalization_4/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_4/moments/mean:output:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_2
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpК
7transformer_block_2/layer_normalization_4/batchnorm/subSubJtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 29
7transformer_block_2/layer_normalization_4/batchnorm/subБ
9transformer_block_2/layer_normalization_4/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_4/batchnorm/add_1
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpМ
7transformer_block_2/sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_2/sequential_2/dense_8/Tensordot/axesУ
7transformer_block_2/sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_2/sequential_2/dense_8/Tensordot/freeс
8transformer_block_2/sequential_2/dense_8/Tensordot/ShapeShape=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_8/Tensordot/ShapeЦ
@transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axis
;transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2GatherV2Atransformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2Ъ
Btransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axisЄ
=transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1GatherV2Atransformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Ktransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1О
8transformer_block_2/sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_2/sequential_2/dense_8/Tensordot/ConstЄ
7transformer_block_2/sequential_2/dense_8/Tensordot/ProdProdDtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Atransformer_block_2/sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_2/sequential_2/dense_8/Tensordot/ProdТ
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_1Ќ
9transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1ProdFtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1:output:0Ctransformer_block_2/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1Т
>transformer_block_2/sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_2/sequential_2/dense_8/Tensordot/concat/axis§
9transformer_block_2/sequential_2/dense_8/Tensordot/concatConcatV2@transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Gtransformer_block_2/sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_8/Tensordot/concatА
8transformer_block_2/sequential_2/dense_8/Tensordot/stackPack@transformer_block_2/sequential_2/dense_8/Tensordot/Prod:output:0Btransformer_block_2/sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_8/Tensordot/stackТ
<transformer_block_2/sequential_2/dense_8/Tensordot/transpose	Transpose=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0Btransformer_block_2/sequential_2/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2>
<transformer_block_2/sequential_2/dense_8/Tensordot/transposeУ
:transformer_block_2/sequential_2/dense_8/Tensordot/ReshapeReshape@transformer_block_2/sequential_2/dense_8/Tensordot/transpose:y:0Atransformer_block_2/sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2<
:transformer_block_2/sequential_2/dense_8/Tensordot/ReshapeТ
9transformer_block_2/sequential_2/dense_8/Tensordot/MatMulMatMulCtransformer_block_2/sequential_2/dense_8/Tensordot/Reshape:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2;
9transformer_block_2/sequential_2/dense_8/Tensordot/MatMulТ
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_2Ц
@transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axis
;transformer_block_2/sequential_2/dense_8/Tensordot/concat_1ConcatV2Dtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Ctransformer_block_2/sequential_2/dense_8/Tensordot/Const_2:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_8/Tensordot/concat_1Д
2transformer_block_2/sequential_2/dense_8/TensordotReshapeCtransformer_block_2/sequential_2/dense_8/Tensordot/MatMul:product:0Dtransformer_block_2/sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 24
2transformer_block_2/sequential_2/dense_8/Tensordot
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpЋ
0transformer_block_2/sequential_2/dense_8/BiasAddBiasAdd;transformer_block_2/sequential_2/dense_8/Tensordot:output:0Gtransformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 22
0transformer_block_2/sequential_2/dense_8/BiasAddз
-transformer_block_2/sequential_2/dense_8/ReluRelu9transformer_block_2/sequential_2/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2/
-transformer_block_2/sequential_2/dense_8/Relu
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpМ
7transformer_block_2/sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_2/sequential_2/dense_9/Tensordot/axesУ
7transformer_block_2/sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_2/sequential_2/dense_9/Tensordot/freeп
8transformer_block_2/sequential_2/dense_9/Tensordot/ShapeShape;transformer_block_2/sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_9/Tensordot/ShapeЦ
@transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axis
;transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2GatherV2Atransformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2Ъ
Btransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axisЄ
=transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1GatherV2Atransformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Ktransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1О
8transformer_block_2/sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_2/sequential_2/dense_9/Tensordot/ConstЄ
7transformer_block_2/sequential_2/dense_9/Tensordot/ProdProdDtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Atransformer_block_2/sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_2/sequential_2/dense_9/Tensordot/ProdТ
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_1Ќ
9transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1ProdFtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1:output:0Ctransformer_block_2/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1Т
>transformer_block_2/sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_2/sequential_2/dense_9/Tensordot/concat/axis§
9transformer_block_2/sequential_2/dense_9/Tensordot/concatConcatV2@transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Gtransformer_block_2/sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_9/Tensordot/concatА
8transformer_block_2/sequential_2/dense_9/Tensordot/stackPack@transformer_block_2/sequential_2/dense_9/Tensordot/Prod:output:0Btransformer_block_2/sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_9/Tensordot/stackР
<transformer_block_2/sequential_2/dense_9/Tensordot/transpose	Transpose;transformer_block_2/sequential_2/dense_8/Relu:activations:0Btransformer_block_2/sequential_2/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2>
<transformer_block_2/sequential_2/dense_9/Tensordot/transposeУ
:transformer_block_2/sequential_2/dense_9/Tensordot/ReshapeReshape@transformer_block_2/sequential_2/dense_9/Tensordot/transpose:y:0Atransformer_block_2/sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2<
:transformer_block_2/sequential_2/dense_9/Tensordot/ReshapeТ
9transformer_block_2/sequential_2/dense_9/Tensordot/MatMulMatMulCtransformer_block_2/sequential_2/dense_9/Tensordot/Reshape:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2;
9transformer_block_2/sequential_2/dense_9/Tensordot/MatMulТ
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_2Ц
@transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axis
;transformer_block_2/sequential_2/dense_9/Tensordot/concat_1ConcatV2Dtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Ctransformer_block_2/sequential_2/dense_9/Tensordot/Const_2:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_9/Tensordot/concat_1Д
2transformer_block_2/sequential_2/dense_9/TensordotReshapeCtransformer_block_2/sequential_2/dense_9/Tensordot/MatMul:product:0Dtransformer_block_2/sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 24
2transformer_block_2/sequential_2/dense_9/Tensordot
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpЋ
0transformer_block_2/sequential_2/dense_9/BiasAddBiasAdd;transformer_block_2/sequential_2/dense_9/Tensordot:output:0Gtransformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 22
0transformer_block_2/sequential_2/dense_9/BiasAdd
+transformer_block_2/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2-
+transformer_block_2/dropout_9/dropout/Const
)transformer_block_2/dropout_9/dropout/MulMul9transformer_block_2/sequential_2/dense_9/BiasAdd:output:04transformer_block_2/dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2+
)transformer_block_2/dropout_9/dropout/MulУ
+transformer_block_2/dropout_9/dropout/ShapeShape9transformer_block_2/sequential_2/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:2-
+transformer_block_2/dropout_9/dropout/Shape
Btransformer_block_2/dropout_9/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_2/dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< *
dtype02D
Btransformer_block_2/dropout_9/dropout/random_uniform/RandomUniformБ
4transformer_block_2/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=26
4transformer_block_2/dropout_9/dropout/GreaterEqual/yК
2transformer_block_2/dropout_9/dropout/GreaterEqualGreaterEqualKtransformer_block_2/dropout_9/dropout/random_uniform/RandomUniform:output:0=transformer_block_2/dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 24
2transformer_block_2/dropout_9/dropout/GreaterEqualн
*transformer_block_2/dropout_9/dropout/CastCast6transformer_block_2/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ< 2,
*transformer_block_2/dropout_9/dropout/Castі
+transformer_block_2/dropout_9/dropout/Mul_1Mul-transformer_block_2/dropout_9/dropout/Mul:z:0.transformer_block_2/dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2-
+transformer_block_2/dropout_9/dropout/Mul_1х
transformer_block_2/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0/transformer_block_2/dropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
transformer_block_2/add_1о
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indicesБ
6transformer_block_2/layer_normalization_5/moments/meanMeantransformer_block_2/add_1:z:0Qtransformer_block_2/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(28
6transformer_block_2/layer_normalization_5/moments/mean
>transformer_block_2/layer_normalization_5/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2@
>transformer_block_2/layer_normalization_5/moments/StopGradientН
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add_1:z:0Gtransformer_block_2/layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2E
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifferenceц
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indicesч
:transformer_block_2/layer_normalization_5/moments/varianceMeanGtransformer_block_2/layer_normalization_5/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2<
:transformer_block_2/layer_normalization_5/moments/varianceЛ
9transformer_block_2/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752;
9transformer_block_2/layer_normalization_5/batchnorm/add/yК
7transformer_block_2/layer_normalization_5/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_5/moments/variance:output:0Btransformer_block_2/layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<29
7transformer_block_2/layer_normalization_5/batchnorm/addђ
9transformer_block_2/layer_normalization_5/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2;
9transformer_block_2/layer_normalization_5/batchnorm/Rsqrt
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpО
7transformer_block_2/layer_normalization_5/batchnorm/mulMul=transformer_block_2/layer_normalization_5/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 29
7transformer_block_2/layer_normalization_5/batchnorm/mul
9transformer_block_2/layer_normalization_5/batchnorm/mul_1Multransformer_block_2/add_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_1Б
9transformer_block_2/layer_normalization_5/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_5/moments/mean:output:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_2
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpК
7transformer_block_2/layer_normalization_5/batchnorm/subSubJtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 29
7transformer_block_2/layer_normalization_5/batchnorm/subБ
9transformer_block_2/layer_normalization_5/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_5/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_5/batchnorm/add_1Ј
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indicesї
global_average_pooling1d_2/MeanMean=transformer_block_2/layer_normalization_5/batchnorm/add_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
global_average_pooling1d_2/Meany
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout_10/dropout/ConstЖ
dropout_10/dropout/MulMul(global_average_pooling1d_2/Mean:output:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_10/dropout/Mul
dropout_10/dropout/ShapeShape(global_average_pooling1d_2/Mean:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shapeе
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype021
/dropout_10/dropout/random_uniform/RandomUniform
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2#
!dropout_10/dropout/GreaterEqual/yъ
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
dropout_10/dropout/GreaterEqual 
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_10/dropout/CastІ
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_10/dropout/Mul_1Ј
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_10/MatMul/ReadVariableOpЄ
dense_10/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_10/MatMulЇ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpЅ
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_10/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout_11/dropout/ConstЉ
dropout_11/dropout/MulMuldense_10/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shapeе
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2#
!dropout_11/dropout/GreaterEqual/yъ
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
dropout_11/dropout/GreaterEqual 
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_11/dropout/CastІ
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_11/dropout/Mul_1Ј
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOpЄ
dense_11/MatMulMatMuldropout_11/dropout/Mul_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_11/MatMulЇ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpЅ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_11/Softmax
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupC^transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpC^transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpO^transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpY^transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpL^transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpD^transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpN^transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpD^transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpN^transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp@^transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpB^transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp@^transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpB^transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2z
;token_and_position_embedding_2/embedding_4/embedding_lookup;token_and_position_embedding_2/embedding_4/embedding_lookup2z
;token_and_position_embedding_2/embedding_5/embedding_lookup;token_and_position_embedding_2/embedding_5/embedding_lookup2
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp2
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp2
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp2
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpNtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp2Д
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpAtransformer_block_2/multi_head_attention_2/key/add/ReadVariableOp2
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpKtransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpCtransformer_block_2/multi_head_attention_2/query/add/ReadVariableOp2
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpMtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpCtransformer_block_2/multi_head_attention_2/value/add/ReadVariableOp2
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpMtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp2
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpAtransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp2
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp2
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpAtransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
ша
Н3
!__inference__traced_restore_10916
file_prefix$
 assignvariableop_dense_10_kernel$
 assignvariableop_1_dense_10_bias&
"assignvariableop_2_dense_11_kernel$
 assignvariableop_3_dense_11_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rateL
Hassignvariableop_9_token_and_position_embedding_2_embedding_4_embeddingsM
Iassignvariableop_10_token_and_position_embedding_2_embedding_5_embeddingsO
Kassignvariableop_11_transformer_block_2_multi_head_attention_2_query_kernelM
Iassignvariableop_12_transformer_block_2_multi_head_attention_2_query_biasM
Iassignvariableop_13_transformer_block_2_multi_head_attention_2_key_kernelK
Gassignvariableop_14_transformer_block_2_multi_head_attention_2_key_biasO
Kassignvariableop_15_transformer_block_2_multi_head_attention_2_value_kernelM
Iassignvariableop_16_transformer_block_2_multi_head_attention_2_value_biasZ
Vassignvariableop_17_transformer_block_2_multi_head_attention_2_attention_output_kernelX
Tassignvariableop_18_transformer_block_2_multi_head_attention_2_attention_output_bias&
"assignvariableop_19_dense_8_kernel$
 assignvariableop_20_dense_8_bias&
"assignvariableop_21_dense_9_kernel$
 assignvariableop_22_dense_9_biasG
Cassignvariableop_23_transformer_block_2_layer_normalization_4_gammaF
Bassignvariableop_24_transformer_block_2_layer_normalization_4_betaG
Cassignvariableop_25_transformer_block_2_layer_normalization_5_gammaF
Bassignvariableop_26_transformer_block_2_layer_normalization_5_beta
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_1.
*assignvariableop_31_adam_dense_10_kernel_m,
(assignvariableop_32_adam_dense_10_bias_m.
*assignvariableop_33_adam_dense_11_kernel_m,
(assignvariableop_34_adam_dense_11_bias_mT
Passignvariableop_35_adam_token_and_position_embedding_2_embedding_4_embeddings_mT
Passignvariableop_36_adam_token_and_position_embedding_2_embedding_5_embeddings_mV
Rassignvariableop_37_adam_transformer_block_2_multi_head_attention_2_query_kernel_mT
Passignvariableop_38_adam_transformer_block_2_multi_head_attention_2_query_bias_mT
Passignvariableop_39_adam_transformer_block_2_multi_head_attention_2_key_kernel_mR
Nassignvariableop_40_adam_transformer_block_2_multi_head_attention_2_key_bias_mV
Rassignvariableop_41_adam_transformer_block_2_multi_head_attention_2_value_kernel_mT
Passignvariableop_42_adam_transformer_block_2_multi_head_attention_2_value_bias_ma
]assignvariableop_43_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_m_
[assignvariableop_44_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_m-
)assignvariableop_45_adam_dense_8_kernel_m+
'assignvariableop_46_adam_dense_8_bias_m-
)assignvariableop_47_adam_dense_9_kernel_m+
'assignvariableop_48_adam_dense_9_bias_mN
Jassignvariableop_49_adam_transformer_block_2_layer_normalization_4_gamma_mM
Iassignvariableop_50_adam_transformer_block_2_layer_normalization_4_beta_mN
Jassignvariableop_51_adam_transformer_block_2_layer_normalization_5_gamma_mM
Iassignvariableop_52_adam_transformer_block_2_layer_normalization_5_beta_m.
*assignvariableop_53_adam_dense_10_kernel_v,
(assignvariableop_54_adam_dense_10_bias_v.
*assignvariableop_55_adam_dense_11_kernel_v,
(assignvariableop_56_adam_dense_11_bias_vT
Passignvariableop_57_adam_token_and_position_embedding_2_embedding_4_embeddings_vT
Passignvariableop_58_adam_token_and_position_embedding_2_embedding_5_embeddings_vV
Rassignvariableop_59_adam_transformer_block_2_multi_head_attention_2_query_kernel_vT
Passignvariableop_60_adam_transformer_block_2_multi_head_attention_2_query_bias_vT
Passignvariableop_61_adam_transformer_block_2_multi_head_attention_2_key_kernel_vR
Nassignvariableop_62_adam_transformer_block_2_multi_head_attention_2_key_bias_vV
Rassignvariableop_63_adam_transformer_block_2_multi_head_attention_2_value_kernel_vT
Passignvariableop_64_adam_transformer_block_2_multi_head_attention_2_value_bias_va
]assignvariableop_65_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_v_
[assignvariableop_66_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_v-
)assignvariableop_67_adam_dense_8_kernel_v+
'assignvariableop_68_adam_dense_8_bias_v-
)assignvariableop_69_adam_dense_9_kernel_v+
'assignvariableop_70_adam_dense_9_bias_vN
Jassignvariableop_71_adam_transformer_block_2_layer_normalization_4_gamma_vM
Iassignvariableop_72_adam_transformer_block_2_layer_normalization_4_beta_vN
Jassignvariableop_73_adam_transformer_block_2_layer_normalization_5_gamma_vM
Iassignvariableop_74_adam_transformer_block_2_layer_normalization_5_beta_v
identity_76ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_8ЂAssignVariableOp_9ь(
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*ј'
valueю'Bы'LB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЉ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
valueЃB LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЊ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4Ё
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѓ
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ѓ
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ђ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Њ
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Э
AssignVariableOp_9AssignVariableOpHassignvariableop_9_token_and_position_embedding_2_embedding_4_embeddingsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10б
AssignVariableOp_10AssignVariableOpIassignvariableop_10_token_and_position_embedding_2_embedding_5_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11г
AssignVariableOp_11AssignVariableOpKassignvariableop_11_transformer_block_2_multi_head_attention_2_query_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12б
AssignVariableOp_12AssignVariableOpIassignvariableop_12_transformer_block_2_multi_head_attention_2_query_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13б
AssignVariableOp_13AssignVariableOpIassignvariableop_13_transformer_block_2_multi_head_attention_2_key_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Я
AssignVariableOp_14AssignVariableOpGassignvariableop_14_transformer_block_2_multi_head_attention_2_key_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15г
AssignVariableOp_15AssignVariableOpKassignvariableop_15_transformer_block_2_multi_head_attention_2_value_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16б
AssignVariableOp_16AssignVariableOpIassignvariableop_16_transformer_block_2_multi_head_attention_2_value_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17о
AssignVariableOp_17AssignVariableOpVassignvariableop_17_transformer_block_2_multi_head_attention_2_attention_output_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18м
AssignVariableOp_18AssignVariableOpTassignvariableop_18_transformer_block_2_multi_head_attention_2_attention_output_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Њ
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_8_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ј
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_8_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Њ
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_9_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ј
AssignVariableOp_22AssignVariableOp assignvariableop_22_dense_9_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ы
AssignVariableOp_23AssignVariableOpCassignvariableop_23_transformer_block_2_layer_normalization_4_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ъ
AssignVariableOp_24AssignVariableOpBassignvariableop_24_transformer_block_2_layer_normalization_4_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ы
AssignVariableOp_25AssignVariableOpCassignvariableop_25_transformer_block_2_layer_normalization_5_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ъ
AssignVariableOp_26AssignVariableOpBassignvariableop_26_transformer_block_2_layer_normalization_5_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ё
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ё
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ѓ
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ѓ
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31В
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_10_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32А
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_10_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33В
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_11_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34А
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_11_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35и
AssignVariableOp_35AssignVariableOpPassignvariableop_35_adam_token_and_position_embedding_2_embedding_4_embeddings_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36и
AssignVariableOp_36AssignVariableOpPassignvariableop_36_adam_token_and_position_embedding_2_embedding_5_embeddings_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37к
AssignVariableOp_37AssignVariableOpRassignvariableop_37_adam_transformer_block_2_multi_head_attention_2_query_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38и
AssignVariableOp_38AssignVariableOpPassignvariableop_38_adam_transformer_block_2_multi_head_attention_2_query_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39и
AssignVariableOp_39AssignVariableOpPassignvariableop_39_adam_transformer_block_2_multi_head_attention_2_key_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ж
AssignVariableOp_40AssignVariableOpNassignvariableop_40_adam_transformer_block_2_multi_head_attention_2_key_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41к
AssignVariableOp_41AssignVariableOpRassignvariableop_41_adam_transformer_block_2_multi_head_attention_2_value_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42и
AssignVariableOp_42AssignVariableOpPassignvariableop_42_adam_transformer_block_2_multi_head_attention_2_value_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43х
AssignVariableOp_43AssignVariableOp]assignvariableop_43_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44у
AssignVariableOp_44AssignVariableOp[assignvariableop_44_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Б
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_8_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Џ
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_8_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Б
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_9_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Џ
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_9_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49в
AssignVariableOp_49AssignVariableOpJassignvariableop_49_adam_transformer_block_2_layer_normalization_4_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50б
AssignVariableOp_50AssignVariableOpIassignvariableop_50_adam_transformer_block_2_layer_normalization_4_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51в
AssignVariableOp_51AssignVariableOpJassignvariableop_51_adam_transformer_block_2_layer_normalization_5_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52б
AssignVariableOp_52AssignVariableOpIassignvariableop_52_adam_transformer_block_2_layer_normalization_5_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53В
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_10_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54А
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_10_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55В
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_11_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56А
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_11_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57и
AssignVariableOp_57AssignVariableOpPassignvariableop_57_adam_token_and_position_embedding_2_embedding_4_embeddings_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58и
AssignVariableOp_58AssignVariableOpPassignvariableop_58_adam_token_and_position_embedding_2_embedding_5_embeddings_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59к
AssignVariableOp_59AssignVariableOpRassignvariableop_59_adam_transformer_block_2_multi_head_attention_2_query_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60и
AssignVariableOp_60AssignVariableOpPassignvariableop_60_adam_transformer_block_2_multi_head_attention_2_query_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61и
AssignVariableOp_61AssignVariableOpPassignvariableop_61_adam_transformer_block_2_multi_head_attention_2_key_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62ж
AssignVariableOp_62AssignVariableOpNassignvariableop_62_adam_transformer_block_2_multi_head_attention_2_key_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63к
AssignVariableOp_63AssignVariableOpRassignvariableop_63_adam_transformer_block_2_multi_head_attention_2_value_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64и
AssignVariableOp_64AssignVariableOpPassignvariableop_64_adam_transformer_block_2_multi_head_attention_2_value_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65х
AssignVariableOp_65AssignVariableOp]assignvariableop_65_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66у
AssignVariableOp_66AssignVariableOp[assignvariableop_66_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Б
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_dense_8_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Џ
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adam_dense_8_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Б
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_dense_9_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Џ
AssignVariableOp_70AssignVariableOp'assignvariableop_70_adam_dense_9_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71в
AssignVariableOp_71AssignVariableOpJassignvariableop_71_adam_transformer_block_2_layer_normalization_4_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72б
AssignVariableOp_72AssignVariableOpIassignvariableop_72_adam_transformer_block_2_layer_normalization_4_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73в
AssignVariableOp_73AssignVariableOpJassignvariableop_73_adam_transformer_block_2_layer_normalization_5_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74б
AssignVariableOp_74AssignVariableOpIassignvariableop_74_adam_transformer_block_2_layer_normalization_5_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_749
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpа
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75У
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*У
_input_shapesБ
Ў: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Я
с
B__inference_dense_9_layer_call_and_return_conditional_losses_10424

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
Е
ё
F__inference_sequential_2_layer_call_and_return_conditional_losses_8318

inputs
dense_8_8307
dense_8_8309
dense_9_8312
dense_9_8314
identityЂdense_8/StatefulPartitionedCallЂdense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_8307dense_8_8309*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_82242!
dense_8/StatefulPartitionedCallЏ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_8312dense_9_8314*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_82702!
dense_9/StatefulPartitionedCallФ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ< ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
ср

A__inference_model_2_layer_call_and_return_conditional_losses_9618

inputsD
@token_and_position_embedding_2_embedding_5_embedding_lookup_9464D
@token_and_position_embedding_2_embedding_4_embedding_lookup_9470Z
Vtransformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_2_multi_head_attention_2_query_add_readvariableop_resourceX
Ttransformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_2_multi_head_attention_2_key_add_readvariableop_resourceZ
Vtransformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_2_multi_head_attention_2_value_add_readvariableop_resourcee
atransformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resourceS
Otransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resourceN
Jtransformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resourceL
Htransformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resourceN
Jtransformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resourceL
Htransformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resourceS
Otransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identityЂdense_10/BiasAdd/ReadVariableOpЂdense_10/MatMul/ReadVariableOpЂdense_11/BiasAdd/ReadVariableOpЂdense_11/MatMul/ReadVariableOpЂ;token_and_position_embedding_2/embedding_4/embedding_lookupЂ;token_and_position_embedding_2/embedding_5/embedding_lookupЂBtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpЂFtransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpЂBtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpЂFtransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpЂNtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpЂXtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpЂAtransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpЂKtransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpЂCtransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpЂMtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpЂCtransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpЂMtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpЂ?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpЂAtransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpЂ?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpЂAtransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp
$token_and_position_embedding_2/ShapeShapeinputs*
T0*
_output_shapes
:2&
$token_and_position_embedding_2/ShapeЛ
2token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ24
2token_and_position_embedding_2/strided_slice/stackЖ
4token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_2/strided_slice/stack_1Ж
4token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_2/strided_slice/stack_2
,token_and_position_embedding_2/strided_sliceStridedSlice-token_and_position_embedding_2/Shape:output:0;token_and_position_embedding_2/strided_slice/stack:output:0=token_and_position_embedding_2/strided_slice/stack_1:output:0=token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_2/strided_slice
*token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_2/range/start
*token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_2/range/delta
$token_and_position_embedding_2/rangeRange3token_and_position_embedding_2/range/start:output:05token_and_position_embedding_2/strided_slice:output:03token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:џџџџџџџџџ2&
$token_and_position_embedding_2/rangeЦ
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGather@token_and_position_embedding_2_embedding_5_embedding_lookup_9464-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*S
_classI
GEloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/9464*'
_output_shapes
:џџџџџџџџџ *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookup
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*S
_classI
GEloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/9464*'
_output_shapes
:џџџџџџџџџ 2F
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2H
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1Г
/token_and_position_embedding_2/embedding_4/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ<21
/token_and_position_embedding_2/embedding_4/Castа
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGather@token_and_position_embedding_2_embedding_4_embedding_lookup_94703token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*S
_classI
GEloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/9470*+
_output_shapes
:џџџџџџџџџ< *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookup
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*S
_classI
GEloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/9470*+
_output_shapes
:џџџџџџџџџ< 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityЁ
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1Љ
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2$
"token_and_position_embedding_2/addЙ
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpщ
>transformer_block_2/multi_head_attention_2/query/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Utransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2@
>transformer_block_2/multi_head_attention_2/query/einsum/Einsum
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpReadVariableOpLtransformer_block_2_multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpХ
4transformer_block_2/multi_head_attention_2/query/addAddV2Gtransformer_block_2/multi_head_attention_2/query/einsum/Einsum:output:0Ktransformer_block_2/multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 26
4transformer_block_2/multi_head_attention_2/query/addГ
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpу
<transformer_block_2/multi_head_attention_2/key/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Stransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2>
<transformer_block_2/multi_head_attention_2/key/einsum/Einsum
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpReadVariableOpJtransformer_block_2_multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpН
2transformer_block_2/multi_head_attention_2/key/addAddV2Etransformer_block_2/multi_head_attention_2/key/einsum/Einsum:output:0Itransformer_block_2/multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 24
2transformer_block_2/multi_head_attention_2/key/addЙ
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpщ
>transformer_block_2/multi_head_attention_2/value/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Utransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2@
>transformer_block_2/multi_head_attention_2/value/einsum/Einsum
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpReadVariableOpLtransformer_block_2_multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpХ
4transformer_block_2/multi_head_attention_2/value/addAddV2Gtransformer_block_2/multi_head_attention_2/value/einsum/Einsum:output:0Ktransformer_block_2/multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 26
4transformer_block_2/multi_head_attention_2/value/addЉ
0transformer_block_2/multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓ5>22
0transformer_block_2/multi_head_attention_2/Mul/y
.transformer_block_2/multi_head_attention_2/MulMul8transformer_block_2/multi_head_attention_2/query/add:z:09transformer_block_2/multi_head_attention_2/Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ< 20
.transformer_block_2/multi_head_attention_2/MulЬ
8transformer_block_2/multi_head_attention_2/einsum/EinsumEinsum6transformer_block_2/multi_head_attention_2/key/add:z:02transformer_block_2/multi_head_attention_2/Mul:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ<<*
equationaecd,abcd->acbe2:
8transformer_block_2/multi_head_attention_2/einsum/Einsum
:transformer_block_2/multi_head_attention_2/softmax/SoftmaxSoftmaxAtransformer_block_2/multi_head_attention_2/einsum/Einsum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2<
:transformer_block_2/multi_head_attention_2/softmax/Softmax
;transformer_block_2/multi_head_attention_2/dropout/IdentityIdentityDtransformer_block_2/multi_head_attention_2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2=
;transformer_block_2/multi_head_attention_2/dropout/Identityф
:transformer_block_2/multi_head_attention_2/einsum_1/EinsumEinsumDtransformer_block_2/multi_head_attention_2/dropout/Identity:output:08transformer_block_2/multi_head_attention_2/value/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationacbe,aecd->abcd2<
:transformer_block_2/multi_head_attention_2/einsum_1/Einsumк
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpЃ
Itransformer_block_2/multi_head_attention_2/attention_output/einsum/EinsumEinsumCtransformer_block_2/multi_head_attention_2/einsum_1/Einsum:output:0`transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ< *
equationabcd,cde->abe2K
Itransformer_block_2/multi_head_attention_2/attention_output/einsum/EinsumД
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpэ
?transformer_block_2/multi_head_attention_2/attention_output/addAddV2Rtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum:output:0Vtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2A
?transformer_block_2/multi_head_attention_2/attention_output/addз
&transformer_block_2/dropout_8/IdentityIdentityCtransformer_block_2/multi_head_attention_2/attention_output/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2(
&transformer_block_2/dropout_8/IdentityЪ
transformer_block_2/addAddV2&token_and_position_embedding_2/add:z:0/transformer_block_2/dropout_8/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
transformer_block_2/addо
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indicesЏ
6transformer_block_2/layer_normalization_4/moments/meanMeantransformer_block_2/add:z:0Qtransformer_block_2/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(28
6transformer_block_2/layer_normalization_4/moments/mean
>transformer_block_2/layer_normalization_4/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2@
>transformer_block_2/layer_normalization_4/moments/StopGradientЛ
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add:z:0Gtransformer_block_2/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2E
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifferenceц
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indicesч
:transformer_block_2/layer_normalization_4/moments/varianceMeanGtransformer_block_2/layer_normalization_4/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2<
:transformer_block_2/layer_normalization_4/moments/varianceЛ
9transformer_block_2/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752;
9transformer_block_2/layer_normalization_4/batchnorm/add/yК
7transformer_block_2/layer_normalization_4/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_4/moments/variance:output:0Btransformer_block_2/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<29
7transformer_block_2/layer_normalization_4/batchnorm/addђ
9transformer_block_2/layer_normalization_4/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2;
9transformer_block_2/layer_normalization_4/batchnorm/Rsqrt
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpО
7transformer_block_2/layer_normalization_4/batchnorm/mulMul=transformer_block_2/layer_normalization_4/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 29
7transformer_block_2/layer_normalization_4/batchnorm/mul
9transformer_block_2/layer_normalization_4/batchnorm/mul_1Multransformer_block_2/add:z:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_1Б
9transformer_block_2/layer_normalization_4/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_4/moments/mean:output:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_2
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpК
7transformer_block_2/layer_normalization_4/batchnorm/subSubJtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 29
7transformer_block_2/layer_normalization_4/batchnorm/subБ
9transformer_block_2/layer_normalization_4/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_4/batchnorm/add_1
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpМ
7transformer_block_2/sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_2/sequential_2/dense_8/Tensordot/axesУ
7transformer_block_2/sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_2/sequential_2/dense_8/Tensordot/freeс
8transformer_block_2/sequential_2/dense_8/Tensordot/ShapeShape=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_8/Tensordot/ShapeЦ
@transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axis
;transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2GatherV2Atransformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2Ъ
Btransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axisЄ
=transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1GatherV2Atransformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Ktransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1О
8transformer_block_2/sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_2/sequential_2/dense_8/Tensordot/ConstЄ
7transformer_block_2/sequential_2/dense_8/Tensordot/ProdProdDtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Atransformer_block_2/sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_2/sequential_2/dense_8/Tensordot/ProdТ
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_1Ќ
9transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1ProdFtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1:output:0Ctransformer_block_2/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1Т
>transformer_block_2/sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_2/sequential_2/dense_8/Tensordot/concat/axis§
9transformer_block_2/sequential_2/dense_8/Tensordot/concatConcatV2@transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Gtransformer_block_2/sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_8/Tensordot/concatА
8transformer_block_2/sequential_2/dense_8/Tensordot/stackPack@transformer_block_2/sequential_2/dense_8/Tensordot/Prod:output:0Btransformer_block_2/sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_8/Tensordot/stackТ
<transformer_block_2/sequential_2/dense_8/Tensordot/transpose	Transpose=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0Btransformer_block_2/sequential_2/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2>
<transformer_block_2/sequential_2/dense_8/Tensordot/transposeУ
:transformer_block_2/sequential_2/dense_8/Tensordot/ReshapeReshape@transformer_block_2/sequential_2/dense_8/Tensordot/transpose:y:0Atransformer_block_2/sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2<
:transformer_block_2/sequential_2/dense_8/Tensordot/ReshapeТ
9transformer_block_2/sequential_2/dense_8/Tensordot/MatMulMatMulCtransformer_block_2/sequential_2/dense_8/Tensordot/Reshape:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2;
9transformer_block_2/sequential_2/dense_8/Tensordot/MatMulТ
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_2Ц
@transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axis
;transformer_block_2/sequential_2/dense_8/Tensordot/concat_1ConcatV2Dtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Ctransformer_block_2/sequential_2/dense_8/Tensordot/Const_2:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_8/Tensordot/concat_1Д
2transformer_block_2/sequential_2/dense_8/TensordotReshapeCtransformer_block_2/sequential_2/dense_8/Tensordot/MatMul:product:0Dtransformer_block_2/sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 24
2transformer_block_2/sequential_2/dense_8/Tensordot
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpЋ
0transformer_block_2/sequential_2/dense_8/BiasAddBiasAdd;transformer_block_2/sequential_2/dense_8/Tensordot:output:0Gtransformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 22
0transformer_block_2/sequential_2/dense_8/BiasAddз
-transformer_block_2/sequential_2/dense_8/ReluRelu9transformer_block_2/sequential_2/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2/
-transformer_block_2/sequential_2/dense_8/Relu
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpМ
7transformer_block_2/sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_2/sequential_2/dense_9/Tensordot/axesУ
7transformer_block_2/sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_2/sequential_2/dense_9/Tensordot/freeп
8transformer_block_2/sequential_2/dense_9/Tensordot/ShapeShape;transformer_block_2/sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_9/Tensordot/ShapeЦ
@transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axis
;transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2GatherV2Atransformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2Ъ
Btransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axisЄ
=transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1GatherV2Atransformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Ktransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1О
8transformer_block_2/sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_2/sequential_2/dense_9/Tensordot/ConstЄ
7transformer_block_2/sequential_2/dense_9/Tensordot/ProdProdDtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Atransformer_block_2/sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_2/sequential_2/dense_9/Tensordot/ProdТ
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_1Ќ
9transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1ProdFtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1:output:0Ctransformer_block_2/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1Т
>transformer_block_2/sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_2/sequential_2/dense_9/Tensordot/concat/axis§
9transformer_block_2/sequential_2/dense_9/Tensordot/concatConcatV2@transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Gtransformer_block_2/sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_9/Tensordot/concatА
8transformer_block_2/sequential_2/dense_9/Tensordot/stackPack@transformer_block_2/sequential_2/dense_9/Tensordot/Prod:output:0Btransformer_block_2/sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_9/Tensordot/stackР
<transformer_block_2/sequential_2/dense_9/Tensordot/transpose	Transpose;transformer_block_2/sequential_2/dense_8/Relu:activations:0Btransformer_block_2/sequential_2/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2>
<transformer_block_2/sequential_2/dense_9/Tensordot/transposeУ
:transformer_block_2/sequential_2/dense_9/Tensordot/ReshapeReshape@transformer_block_2/sequential_2/dense_9/Tensordot/transpose:y:0Atransformer_block_2/sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2<
:transformer_block_2/sequential_2/dense_9/Tensordot/ReshapeТ
9transformer_block_2/sequential_2/dense_9/Tensordot/MatMulMatMulCtransformer_block_2/sequential_2/dense_9/Tensordot/Reshape:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2;
9transformer_block_2/sequential_2/dense_9/Tensordot/MatMulТ
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_2Ц
@transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axis
;transformer_block_2/sequential_2/dense_9/Tensordot/concat_1ConcatV2Dtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Ctransformer_block_2/sequential_2/dense_9/Tensordot/Const_2:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_9/Tensordot/concat_1Д
2transformer_block_2/sequential_2/dense_9/TensordotReshapeCtransformer_block_2/sequential_2/dense_9/Tensordot/MatMul:product:0Dtransformer_block_2/sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 24
2transformer_block_2/sequential_2/dense_9/Tensordot
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpЋ
0transformer_block_2/sequential_2/dense_9/BiasAddBiasAdd;transformer_block_2/sequential_2/dense_9/Tensordot:output:0Gtransformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 22
0transformer_block_2/sequential_2/dense_9/BiasAddЭ
&transformer_block_2/dropout_9/IdentityIdentity9transformer_block_2/sequential_2/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2(
&transformer_block_2/dropout_9/Identityх
transformer_block_2/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0/transformer_block_2/dropout_9/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
transformer_block_2/add_1о
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indicesБ
6transformer_block_2/layer_normalization_5/moments/meanMeantransformer_block_2/add_1:z:0Qtransformer_block_2/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(28
6transformer_block_2/layer_normalization_5/moments/mean
>transformer_block_2/layer_normalization_5/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2@
>transformer_block_2/layer_normalization_5/moments/StopGradientН
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add_1:z:0Gtransformer_block_2/layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2E
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifferenceц
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indicesч
:transformer_block_2/layer_normalization_5/moments/varianceMeanGtransformer_block_2/layer_normalization_5/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2<
:transformer_block_2/layer_normalization_5/moments/varianceЛ
9transformer_block_2/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752;
9transformer_block_2/layer_normalization_5/batchnorm/add/yК
7transformer_block_2/layer_normalization_5/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_5/moments/variance:output:0Btransformer_block_2/layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<29
7transformer_block_2/layer_normalization_5/batchnorm/addђ
9transformer_block_2/layer_normalization_5/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2;
9transformer_block_2/layer_normalization_5/batchnorm/Rsqrt
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpО
7transformer_block_2/layer_normalization_5/batchnorm/mulMul=transformer_block_2/layer_normalization_5/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 29
7transformer_block_2/layer_normalization_5/batchnorm/mul
9transformer_block_2/layer_normalization_5/batchnorm/mul_1Multransformer_block_2/add_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_1Б
9transformer_block_2/layer_normalization_5/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_5/moments/mean:output:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_2
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpК
7transformer_block_2/layer_normalization_5/batchnorm/subSubJtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 29
7transformer_block_2/layer_normalization_5/batchnorm/subБ
9transformer_block_2/layer_normalization_5/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_5/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2;
9transformer_block_2/layer_normalization_5/batchnorm/add_1Ј
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indicesї
global_average_pooling1d_2/MeanMean=transformer_block_2/layer_normalization_5/batchnorm/add_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
global_average_pooling1d_2/Mean
dropout_10/IdentityIdentity(global_average_pooling1d_2/Mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_10/IdentityЈ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_10/MatMul/ReadVariableOpЄ
dense_10/MatMulMatMuldropout_10/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_10/MatMulЇ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpЅ
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_10/Relu
dropout_11/IdentityIdentitydense_10/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_11/IdentityЈ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOpЄ
dense_11/MatMulMatMuldropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_11/MatMulЇ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpЅ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_11/Softmax
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupC^transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpC^transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpO^transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpY^transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpL^transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpD^transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpN^transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpD^transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpN^transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp@^transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpB^transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp@^transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpB^transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2z
;token_and_position_embedding_2/embedding_4/embedding_lookup;token_and_position_embedding_2/embedding_4/embedding_lookup2z
;token_and_position_embedding_2/embedding_5/embedding_lookup;token_and_position_embedding_2/embedding_5/embedding_lookup2
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp2
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp2
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp2
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpNtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp2Д
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpAtransformer_block_2/multi_head_attention_2/key/add/ReadVariableOp2
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpKtransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpCtransformer_block_2/multi_head_attention_2/query/add/ReadVariableOp2
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpMtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpCtransformer_block_2/multi_head_attention_2/value/add/ReadVariableOp2
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpMtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp2
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpAtransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp2
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp2
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpAtransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
иH
І
G__inference_sequential_2_layer_call_and_return_conditional_losses_10271

inputs-
)dense_8_tensordot_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityЂdense_8/BiasAdd/ReadVariableOpЂ dense_8/Tensordot/ReadVariableOpЂdense_9/BiasAdd/ReadVariableOpЂ dense_9/Tensordot/ReadVariableOpЎ
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_8/Tensordot/ReadVariableOpz
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/axes
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_8/Tensordot/freeh
dense_8/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_8/Tensordot/Shape
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/GatherV2/axisљ
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_8/Tensordot/GatherV2_1/axisџ
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2_1|
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const 
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const_1Ј
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod_1
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_8/Tensordot/concat/axisи
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concatЌ
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/stackЈ
dense_8/Tensordot/transpose	Transposeinputs!dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_8/Tensordot/transposeП
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_8/Tensordot/ReshapeО
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_8/Tensordot/MatMul
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const_2
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/concat_1/axisх
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat_1А
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_8/TensordotЄ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_8/BiasAdd/ReadVariableOpЇ
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_8/BiasAddt
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_8/ReluЎ
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axes
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/free|
dense_9/Tensordot/ShapeShapedense_8/Relu:activations:0*
T0*
_output_shapes
:2
dense_9/Tensordot/Shape
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisљ
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axisџ
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1Ј
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisи
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatЌ
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stackМ
dense_9/Tensordot/transpose	Transposedense_8/Relu:activations:0!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_9/Tensordot/transposeП
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_9/Tensordot/ReshapeО
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_9/Tensordot/MatMul
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_2
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisх
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1А
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_9/TensordotЄ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_9/BiasAdd/ReadVariableOpЇ
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_9/BiasAddј
IdentityIdentitydense_9/BiasAdd:output:0^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ< ::::2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
ч
|
'__inference_dense_9_layer_call_fn_10433

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_82702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
Ч
b
D__inference_dropout_10_layer_call_and_return_conditional_losses_8832

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

F
*__inference_dropout_11_layer_call_fn_10194

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_88892
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
иH
І
G__inference_sequential_2_layer_call_and_return_conditional_losses_10328

inputs-
)dense_8_tensordot_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityЂdense_8/BiasAdd/ReadVariableOpЂ dense_8/Tensordot/ReadVariableOpЂdense_9/BiasAdd/ReadVariableOpЂ dense_9/Tensordot/ReadVariableOpЎ
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_8/Tensordot/ReadVariableOpz
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/axes
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_8/Tensordot/freeh
dense_8/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_8/Tensordot/Shape
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/GatherV2/axisљ
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_8/Tensordot/GatherV2_1/axisџ
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2_1|
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const 
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const_1Ј
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod_1
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_8/Tensordot/concat/axisи
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concatЌ
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/stackЈ
dense_8/Tensordot/transpose	Transposeinputs!dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_8/Tensordot/transposeП
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_8/Tensordot/ReshapeО
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_8/Tensordot/MatMul
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const_2
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/concat_1/axisх
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat_1А
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_8/TensordotЄ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_8/BiasAdd/ReadVariableOpЇ
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_8/BiasAddt
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_8/ReluЎ
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axes
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/free|
dense_9/Tensordot/ShapeShapedense_8/Relu:activations:0*
T0*
_output_shapes
:2
dense_9/Tensordot/Shape
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisљ
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axisџ
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1Ј
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisи
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatЌ
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stackМ
dense_9/Tensordot/transpose	Transposedense_8/Relu:activations:0!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_9/Tensordot/transposeП
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_9/Tensordot/ReshapeО
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_9/Tensordot/MatMul
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_2
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisх
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1А
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_9/TensordotЄ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_9/BiasAdd/ReadVariableOpЇ
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dense_9/BiasAddј
IdentityIdentitydense_9/BiasAdd:output:0^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ< ::::2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
В

,__inference_sequential_2_layer_call_fn_10354

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_83452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ< ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
Г)

A__inference_model_2_layer_call_and_return_conditional_losses_8985
input_4'
#token_and_position_embedding_2_8933'
#token_and_position_embedding_2_8935
transformer_block_2_8938
transformer_block_2_8940
transformer_block_2_8942
transformer_block_2_8944
transformer_block_2_8946
transformer_block_2_8948
transformer_block_2_8950
transformer_block_2_8952
transformer_block_2_8954
transformer_block_2_8956
transformer_block_2_8958
transformer_block_2_8960
transformer_block_2_8962
transformer_block_2_8964
transformer_block_2_8966
transformer_block_2_8968
dense_10_8973
dense_10_8975
dense_11_8979
dense_11_8981
identityЂ dense_10/StatefulPartitionedCallЂ dense_11/StatefulPartitionedCallЂ6token_and_position_embedding_2/StatefulPartitionedCallЂ+transformer_block_2/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_4#token_and_position_embedding_2_8933#token_and_position_embedding_2_8935*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_840328
6token_and_position_embedding_2/StatefulPartitionedCall
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_8938transformer_block_2_8940transformer_block_2_8942transformer_block_2_8944transformer_block_2_8946transformer_block_2_8948transformer_block_2_8950transformer_block_2_8952transformer_block_2_8954transformer_block_2_8956transformer_block_2_8958transformer_block_2_8960transformer_block_2_8962transformer_block_2_8964transformer_block_2_8966transformer_block_2_8968*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_86942-
+transformer_block_2/StatefulPartitionedCallЖ
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_88082,
*global_average_pooling1d_2/PartitionedCall
dropout_10/PartitionedCallPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_88322
dropout_10/PartitionedCallЋ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_10_8973dense_10_8975*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_88562"
 dense_10/StatefulPartitionedCallћ
dropout_11/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_88892
dropout_11/PartitionedCallЋ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_11_8979dense_11_8981*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_89132"
 dense_11/StatefulPartitionedCallЊ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ<
!
_user_specified_name	input_4
Ш
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_10184

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь	
л
B__inference_dense_10_layer_call_and_return_conditional_losses_8856

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Р,
ж
A__inference_model_2_layer_call_and_return_conditional_losses_8930
input_4'
#token_and_position_embedding_2_8414'
#token_and_position_embedding_2_8416
transformer_block_2_8770
transformer_block_2_8772
transformer_block_2_8774
transformer_block_2_8776
transformer_block_2_8778
transformer_block_2_8780
transformer_block_2_8782
transformer_block_2_8784
transformer_block_2_8786
transformer_block_2_8788
transformer_block_2_8790
transformer_block_2_8792
transformer_block_2_8794
transformer_block_2_8796
transformer_block_2_8798
transformer_block_2_8800
dense_10_8867
dense_10_8869
dense_11_8924
dense_11_8926
identityЂ dense_10/StatefulPartitionedCallЂ dense_11/StatefulPartitionedCallЂ"dropout_10/StatefulPartitionedCallЂ"dropout_11/StatefulPartitionedCallЂ6token_and_position_embedding_2/StatefulPartitionedCallЂ+transformer_block_2/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_4#token_and_position_embedding_2_8414#token_and_position_embedding_2_8416*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_840328
6token_and_position_embedding_2/StatefulPartitionedCall
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_8770transformer_block_2_8772transformer_block_2_8774transformer_block_2_8776transformer_block_2_8778transformer_block_2_8780transformer_block_2_8782transformer_block_2_8784transformer_block_2_8786transformer_block_2_8788transformer_block_2_8790transformer_block_2_8792transformer_block_2_8794transformer_block_2_8796transformer_block_2_8798transformer_block_2_8800*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_85672-
+transformer_block_2/StatefulPartitionedCallЖ
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_88082,
*global_average_pooling1d_2/PartitionedCall
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_88272$
"dropout_10/StatefulPartitionedCallГ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_10_8867dense_10_8869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_88562"
 dense_10/StatefulPartitionedCallИ
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_88842$
"dropout_11/StatefulPartitionedCallГ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_11_8924dense_11_8926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_89132"
 dense_11/StatefulPartitionedCallє
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ<
!
_user_specified_name	input_4
Џ 
с
B__inference_dense_8_layer_call_and_return_conditional_losses_10385

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
Н,
е
A__inference_model_2_layer_call_and_return_conditional_losses_9043

inputs'
#token_and_position_embedding_2_8991'
#token_and_position_embedding_2_8993
transformer_block_2_8996
transformer_block_2_8998
transformer_block_2_9000
transformer_block_2_9002
transformer_block_2_9004
transformer_block_2_9006
transformer_block_2_9008
transformer_block_2_9010
transformer_block_2_9012
transformer_block_2_9014
transformer_block_2_9016
transformer_block_2_9018
transformer_block_2_9020
transformer_block_2_9022
transformer_block_2_9024
transformer_block_2_9026
dense_10_9031
dense_10_9033
dense_11_9037
dense_11_9039
identityЂ dense_10/StatefulPartitionedCallЂ dense_11/StatefulPartitionedCallЂ"dropout_10/StatefulPartitionedCallЂ"dropout_11/StatefulPartitionedCallЂ6token_and_position_embedding_2/StatefulPartitionedCallЂ+transformer_block_2/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs#token_and_position_embedding_2_8991#token_and_position_embedding_2_8993*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_840328
6token_and_position_embedding_2/StatefulPartitionedCall
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_8996transformer_block_2_8998transformer_block_2_9000transformer_block_2_9002transformer_block_2_9004transformer_block_2_9006transformer_block_2_9008transformer_block_2_9010transformer_block_2_9012transformer_block_2_9014transformer_block_2_9016transformer_block_2_9018transformer_block_2_9020transformer_block_2_9022transformer_block_2_9024transformer_block_2_9026*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_85672-
+transformer_block_2/StatefulPartitionedCallЖ
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_88082,
*global_average_pooling1d_2/PartitionedCall
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_88272$
"dropout_10/StatefulPartitionedCallГ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_10_9031dense_10_9033*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_88562"
 dense_10/StatefulPartitionedCallИ
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_88842$
"dropout_11/StatefulPartitionedCallГ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_11_9037dense_11_9039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_89132"
 dense_11/StatefulPartitionedCallє
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs

c
*__inference_dropout_11_layer_call_fn_10189

inputs
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_88842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч
|
'__inference_dense_8_layer_call_fn_10394

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_82242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
к
ў
X__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_8403
x%
!embedding_5_embedding_lookup_8390%
!embedding_4_embedding_lookup_8396
identityЂembedding_4/embedding_lookupЂembedding_5/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:џџџџџџџџџ2
rangeЋ
embedding_5/embedding_lookupResourceGather!embedding_5_embedding_lookup_8390range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_5/embedding_lookup/8390*'
_output_shapes
:џџџџџџџџџ *
dtype02
embedding_5/embedding_lookup
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_5/embedding_lookup/8390*'
_output_shapes
:џџџџџџџџџ 2'
%embedding_5/embedding_lookup/IdentityР
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'embedding_5/embedding_lookup/Identity_1p
embedding_4/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ<2
embedding_4/CastЕ
embedding_4/embedding_lookupResourceGather!embedding_4_embedding_lookup_8396embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_4/embedding_lookup/8396*+
_output_shapes
:џџџџџџџџџ< *
dtype02
embedding_4/embedding_lookup
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_4/embedding_lookup/8396*+
_output_shapes
:џџџџџџџџџ< 2'
%embedding_4/embedding_lookup/IdentityФ
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2)
'embedding_4/embedding_lookup/Identity_1­
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
add
IdentityIdentityadd:z:0^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ<::2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:J F
'
_output_shapes
:џџџџџџџџџ<

_user_specified_namex
Ю
р
A__inference_dense_9_layer_call_and_return_conditional_losses_8270

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
џл
а
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_8694

inputsF
Bmulti_head_attention_2_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_query_add_readvariableop_resourceD
@multi_head_attention_2_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_2_key_add_readvariableop_resourceF
Bmulti_head_attention_2_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_value_add_readvariableop_resourceQ
Mmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_2_attention_output_add_readvariableop_resource?
;layer_normalization_4_batchnorm_mul_readvariableop_resource;
7layer_normalization_4_batchnorm_readvariableop_resource:
6sequential_2_dense_8_tensordot_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resource:
6sequential_2_dense_9_tensordot_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resource?
;layer_normalization_5_batchnorm_mul_readvariableop_resource;
7layer_normalization_5_batchnorm_readvariableop_resource
identityЂ.layer_normalization_4/batchnorm/ReadVariableOpЂ2layer_normalization_4/batchnorm/mul/ReadVariableOpЂ.layer_normalization_5/batchnorm/ReadVariableOpЂ2layer_normalization_5/batchnorm/mul/ReadVariableOpЂ:multi_head_attention_2/attention_output/add/ReadVariableOpЂDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpЂ-multi_head_attention_2/key/add/ReadVariableOpЂ7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpЂ/multi_head_attention_2/query/add/ReadVariableOpЂ9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpЂ/multi_head_attention_2/value/add/ReadVariableOpЂ9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpЂ+sequential_2/dense_8/BiasAdd/ReadVariableOpЂ-sequential_2/dense_8/Tensordot/ReadVariableOpЂ+sequential_2/dense_9/BiasAdd/ReadVariableOpЂ-sequential_2/dense_9/Tensordot/ReadVariableOp§
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_2/query/einsum/EinsumEinsuminputsAmulti_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2,
*multi_head_attention_2/query/einsum/Einsumл
/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp8multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_2/query/add/ReadVariableOpѕ
 multi_head_attention_2/query/addAddV23multi_head_attention_2/query/einsum/Einsum:output:07multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2"
 multi_head_attention_2/query/addї
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_2/key/einsum/EinsumEinsuminputs?multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2*
(multi_head_attention_2/key/einsum/Einsumе
-multi_head_attention_2/key/add/ReadVariableOpReadVariableOp6multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_2/key/add/ReadVariableOpэ
multi_head_attention_2/key/addAddV21multi_head_attention_2/key/einsum/Einsum:output:05multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2 
multi_head_attention_2/key/add§
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_2/value/einsum/EinsumEinsuminputsAmulti_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2,
*multi_head_attention_2/value/einsum/Einsumл
/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp8multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_2/value/add/ReadVariableOpѕ
 multi_head_attention_2/value/addAddV23multi_head_attention_2/value/einsum/Einsum:output:07multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2"
 multi_head_attention_2/value/add
multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓ5>2
multi_head_attention_2/Mul/yЦ
multi_head_attention_2/MulMul$multi_head_attention_2/query/add:z:0%multi_head_attention_2/Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2
multi_head_attention_2/Mulќ
$multi_head_attention_2/einsum/EinsumEinsum"multi_head_attention_2/key/add:z:0multi_head_attention_2/Mul:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ<<*
equationaecd,abcd->acbe2&
$multi_head_attention_2/einsum/EinsumФ
&multi_head_attention_2/softmax/SoftmaxSoftmax-multi_head_attention_2/einsum/Einsum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2(
&multi_head_attention_2/softmax/SoftmaxЪ
'multi_head_attention_2/dropout/IdentityIdentity0multi_head_attention_2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2)
'multi_head_attention_2/dropout/Identity
&multi_head_attention_2/einsum_1/EinsumEinsum0multi_head_attention_2/dropout/Identity:output:0$multi_head_attention_2/value/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationacbe,aecd->abcd2(
&multi_head_attention_2/einsum_1/Einsum
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpг
5multi_head_attention_2/attention_output/einsum/EinsumEinsum/multi_head_attention_2/einsum_1/Einsum:output:0Lmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ< *
equationabcd,cde->abe27
5multi_head_attention_2/attention_output/einsum/Einsumј
:multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_2/attention_output/add/ReadVariableOp
+multi_head_attention_2/attention_output/addAddV2>multi_head_attention_2/attention_output/einsum/Einsum:output:0Bmulti_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2-
+multi_head_attention_2/attention_output/add
dropout_8/IdentityIdentity/multi_head_attention_2/attention_output/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_8/Identityn
addAddV2inputsdropout_8/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
addЖ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesп
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2$
"layer_normalization_4/moments/meanЫ
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2,
*layer_normalization_4/moments/StopGradientы
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 21
/layer_normalization_4/moments/SquaredDifferenceО
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indices
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2(
&layer_normalization_4/moments/variance
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752'
%layer_normalization_4/batchnorm/add/yъ
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2%
#layer_normalization_4/batchnorm/addЖ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2'
%layer_normalization_4/batchnorm/Rsqrtр
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpю
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_4/batchnorm/mulН
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/mul_1с
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/mul_2д
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpъ
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_4/batchnorm/subс
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/add_1е
-sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_2/dense_8/Tensordot/ReadVariableOp
#sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_8/Tensordot/axes
#sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_8/Tensordot/freeЅ
$sequential_2/dense_8/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/Shape
,sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/GatherV2/axisК
'sequential_2/dense_8/Tensordot/GatherV2GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/free:output:05sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/GatherV2Ђ
.sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_8/Tensordot/GatherV2_1/axisР
)sequential_2/dense_8/Tensordot/GatherV2_1GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/axes:output:07sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_8/Tensordot/GatherV2_1
$sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_8/Tensordot/Constд
#sequential_2/dense_8/Tensordot/ProdProd0sequential_2/dense_8/Tensordot/GatherV2:output:0-sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_8/Tensordot/Prod
&sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_1м
%sequential_2/dense_8/Tensordot/Prod_1Prod2sequential_2/dense_8/Tensordot/GatherV2_1:output:0/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_8/Tensordot/Prod_1
*sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_8/Tensordot/concat/axis
%sequential_2/dense_8/Tensordot/concatConcatV2,sequential_2/dense_8/Tensordot/free:output:0,sequential_2/dense_8/Tensordot/axes:output:03sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_8/Tensordot/concatр
$sequential_2/dense_8/Tensordot/stackPack,sequential_2/dense_8/Tensordot/Prod:output:0.sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/stackђ
(sequential_2/dense_8/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0.sequential_2/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2*
(sequential_2/dense_8/Tensordot/transposeѓ
&sequential_2/dense_8/Tensordot/ReshapeReshape,sequential_2/dense_8/Tensordot/transpose:y:0-sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2(
&sequential_2/dense_8/Tensordot/Reshapeђ
%sequential_2/dense_8/Tensordot/MatMulMatMul/sequential_2/dense_8/Tensordot/Reshape:output:05sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_2/dense_8/Tensordot/MatMul
&sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_2
,sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/concat_1/axisІ
'sequential_2/dense_8/Tensordot/concat_1ConcatV20sequential_2/dense_8/Tensordot/GatherV2:output:0/sequential_2/dense_8/Tensordot/Const_2:output:05sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/concat_1ф
sequential_2/dense_8/TensordotReshape/sequential_2/dense_8/Tensordot/MatMul:product:00sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
sequential_2/dense_8/TensordotЫ
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOpл
sequential_2/dense_8/BiasAddBiasAdd'sequential_2/dense_8/Tensordot:output:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_8/BiasAdd
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_8/Reluе
-sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_2/dense_9/Tensordot/ReadVariableOp
#sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_9/Tensordot/axes
#sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_9/Tensordot/freeЃ
$sequential_2/dense_9/Tensordot/ShapeShape'sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/Shape
,sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/GatherV2/axisК
'sequential_2/dense_9/Tensordot/GatherV2GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/free:output:05sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/GatherV2Ђ
.sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_9/Tensordot/GatherV2_1/axisР
)sequential_2/dense_9/Tensordot/GatherV2_1GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/axes:output:07sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_9/Tensordot/GatherV2_1
$sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_9/Tensordot/Constд
#sequential_2/dense_9/Tensordot/ProdProd0sequential_2/dense_9/Tensordot/GatherV2:output:0-sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_9/Tensordot/Prod
&sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_1м
%sequential_2/dense_9/Tensordot/Prod_1Prod2sequential_2/dense_9/Tensordot/GatherV2_1:output:0/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_9/Tensordot/Prod_1
*sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_9/Tensordot/concat/axis
%sequential_2/dense_9/Tensordot/concatConcatV2,sequential_2/dense_9/Tensordot/free:output:0,sequential_2/dense_9/Tensordot/axes:output:03sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_9/Tensordot/concatр
$sequential_2/dense_9/Tensordot/stackPack,sequential_2/dense_9/Tensordot/Prod:output:0.sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/stack№
(sequential_2/dense_9/Tensordot/transpose	Transpose'sequential_2/dense_8/Relu:activations:0.sequential_2/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2*
(sequential_2/dense_9/Tensordot/transposeѓ
&sequential_2/dense_9/Tensordot/ReshapeReshape,sequential_2/dense_9/Tensordot/transpose:y:0-sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2(
&sequential_2/dense_9/Tensordot/Reshapeђ
%sequential_2/dense_9/Tensordot/MatMulMatMul/sequential_2/dense_9/Tensordot/Reshape:output:05sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_2/dense_9/Tensordot/MatMul
&sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_2
,sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/concat_1/axisІ
'sequential_2/dense_9/Tensordot/concat_1ConcatV20sequential_2/dense_9/Tensordot/GatherV2:output:0/sequential_2/dense_9/Tensordot/Const_2:output:05sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/concat_1ф
sequential_2/dense_9/TensordotReshape/sequential_2/dense_9/Tensordot/MatMul:product:00sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
sequential_2/dense_9/TensordotЫ
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOpл
sequential_2/dense_9/BiasAddBiasAdd'sequential_2/dense_9/Tensordot:output:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_9/BiasAdd
dropout_9/IdentityIdentity%sequential_2/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_9/Identity
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_9/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
add_1Ж
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indicesс
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2$
"layer_normalization_5/moments/meanЫ
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2,
*layer_normalization_5/moments/StopGradientэ
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 21
/layer_normalization_5/moments/SquaredDifferenceО
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indices
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2(
&layer_normalization_5/moments/variance
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752'
%layer_normalization_5/batchnorm/add/yъ
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2%
#layer_normalization_5/batchnorm/addЖ
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2'
%layer_normalization_5/batchnorm/Rsqrtр
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOpю
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_5/batchnorm/mulП
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/mul_1с
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/mul_2д
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOpъ
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_5/batchnorm/subс
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/add_1г
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp;^multi_head_attention_2/attention_output/add/ReadVariableOpE^multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_2/key/add/ReadVariableOp8^multi_head_attention_2/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/query/add/ReadVariableOp:^multi_head_attention_2/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/value/add/ReadVariableOp:^multi_head_attention_2/value/einsum/Einsum/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp.^sequential_2/dense_8/Tensordot/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp.^sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:џџџџџџџџџ< ::::::::::::::::2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_2/attention_output/add/ReadVariableOp:multi_head_attention_2/attention_output/add/ReadVariableOp2
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_2/key/add/ReadVariableOp-multi_head_attention_2/key/add/ReadVariableOp2r
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/query/add/ReadVariableOp/multi_head_attention_2/query/add/ReadVariableOp2v
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/value/add/ReadVariableOp/multi_head_attention_2/value/add/ReadVariableOp2v
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2^
-sequential_2/dense_8/Tensordot/ReadVariableOp-sequential_2/dense_8/Tensordot/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2^
-sequential_2/dense_9/Tensordot/ReadVariableOp-sequential_2/dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
й
}
(__inference_dense_11_layer_call_fn_10214

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_89132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_10104

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц
Ѕ
+__inference_sequential_2_layer_call_fn_8329
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_83182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ< ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ< 
'
_user_specified_namedense_8_input
ў

=__inference_token_and_position_embedding_2_layer_call_fn_9749
x
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_84032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ<::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:џџџџџџџџџ<

_user_specified_namex

c
*__inference_dropout_10_layer_call_fn_10142

inputs
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_88272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
є	
л
B__inference_dense_11_layer_call_and_return_conditional_losses_8913

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­ќ
а
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_8567

inputsF
Bmulti_head_attention_2_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_query_add_readvariableop_resourceD
@multi_head_attention_2_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_2_key_add_readvariableop_resourceF
Bmulti_head_attention_2_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_value_add_readvariableop_resourceQ
Mmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_2_attention_output_add_readvariableop_resource?
;layer_normalization_4_batchnorm_mul_readvariableop_resource;
7layer_normalization_4_batchnorm_readvariableop_resource:
6sequential_2_dense_8_tensordot_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resource:
6sequential_2_dense_9_tensordot_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resource?
;layer_normalization_5_batchnorm_mul_readvariableop_resource;
7layer_normalization_5_batchnorm_readvariableop_resource
identityЂ.layer_normalization_4/batchnorm/ReadVariableOpЂ2layer_normalization_4/batchnorm/mul/ReadVariableOpЂ.layer_normalization_5/batchnorm/ReadVariableOpЂ2layer_normalization_5/batchnorm/mul/ReadVariableOpЂ:multi_head_attention_2/attention_output/add/ReadVariableOpЂDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpЂ-multi_head_attention_2/key/add/ReadVariableOpЂ7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpЂ/multi_head_attention_2/query/add/ReadVariableOpЂ9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpЂ/multi_head_attention_2/value/add/ReadVariableOpЂ9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpЂ+sequential_2/dense_8/BiasAdd/ReadVariableOpЂ-sequential_2/dense_8/Tensordot/ReadVariableOpЂ+sequential_2/dense_9/BiasAdd/ReadVariableOpЂ-sequential_2/dense_9/Tensordot/ReadVariableOp§
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_2/query/einsum/EinsumEinsuminputsAmulti_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2,
*multi_head_attention_2/query/einsum/Einsumл
/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp8multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_2/query/add/ReadVariableOpѕ
 multi_head_attention_2/query/addAddV23multi_head_attention_2/query/einsum/Einsum:output:07multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2"
 multi_head_attention_2/query/addї
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_2/key/einsum/EinsumEinsuminputs?multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2*
(multi_head_attention_2/key/einsum/Einsumе
-multi_head_attention_2/key/add/ReadVariableOpReadVariableOp6multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_2/key/add/ReadVariableOpэ
multi_head_attention_2/key/addAddV21multi_head_attention_2/key/einsum/Einsum:output:05multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2 
multi_head_attention_2/key/add§
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_2/value/einsum/EinsumEinsuminputsAmulti_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2,
*multi_head_attention_2/value/einsum/Einsumл
/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp8multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_2/value/add/ReadVariableOpѕ
 multi_head_attention_2/value/addAddV23multi_head_attention_2/value/einsum/Einsum:output:07multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2"
 multi_head_attention_2/value/add
multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓ5>2
multi_head_attention_2/Mul/yЦ
multi_head_attention_2/MulMul$multi_head_attention_2/query/add:z:0%multi_head_attention_2/Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2
multi_head_attention_2/Mulќ
$multi_head_attention_2/einsum/EinsumEinsum"multi_head_attention_2/key/add:z:0multi_head_attention_2/Mul:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ<<*
equationaecd,abcd->acbe2&
$multi_head_attention_2/einsum/EinsumФ
&multi_head_attention_2/softmax/SoftmaxSoftmax-multi_head_attention_2/einsum/Einsum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2(
&multi_head_attention_2/softmax/SoftmaxЁ
,multi_head_attention_2/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_2/dropout/dropout/Const
*multi_head_attention_2/dropout/dropout/MulMul0multi_head_attention_2/softmax/Softmax:softmax:05multi_head_attention_2/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2,
*multi_head_attention_2/dropout/dropout/MulМ
,multi_head_attention_2/dropout/dropout/ShapeShape0multi_head_attention_2/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_2/dropout/dropout/Shape
Cmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_2/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<*
dtype02E
Cmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniformГ
5multi_head_attention_2/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_2/dropout/dropout/GreaterEqual/yТ
3multi_head_attention_2/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_2/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<25
3multi_head_attention_2/dropout/dropout/GreaterEqualф
+multi_head_attention_2/dropout/dropout/CastCast7multi_head_attention_2/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ<<2-
+multi_head_attention_2/dropout/dropout/Castў
,multi_head_attention_2/dropout/dropout/Mul_1Mul.multi_head_attention_2/dropout/dropout/Mul:z:0/multi_head_attention_2/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2.
,multi_head_attention_2/dropout/dropout/Mul_1
&multi_head_attention_2/einsum_1/EinsumEinsum0multi_head_attention_2/dropout/dropout/Mul_1:z:0$multi_head_attention_2/value/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationacbe,aecd->abcd2(
&multi_head_attention_2/einsum_1/Einsum
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpг
5multi_head_attention_2/attention_output/einsum/EinsumEinsum/multi_head_attention_2/einsum_1/Einsum:output:0Lmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ< *
equationabcd,cde->abe27
5multi_head_attention_2/attention_output/einsum/Einsumј
:multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_2/attention_output/add/ReadVariableOp
+multi_head_attention_2/attention_output/addAddV2>multi_head_attention_2/attention_output/einsum/Einsum:output:0Bmulti_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2-
+multi_head_attention_2/attention_output/addw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout_8/dropout/ConstО
dropout_8/dropout/MulMul/multi_head_attention_2/attention_output/add:z:0 dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_8/dropout/Mul
dropout_8/dropout/ShapeShape/multi_head_attention_2/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shapeж
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< *
dtype020
.dropout_8/dropout/random_uniform/RandomUniform
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2"
 dropout_8/dropout/GreaterEqual/yъ
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
dropout_8/dropout/GreaterEqualЁ
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ< 2
dropout_8/dropout/CastІ
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_8/dropout/Mul_1n
addAddV2inputsdropout_8/dropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
addЖ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesп
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2$
"layer_normalization_4/moments/meanЫ
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2,
*layer_normalization_4/moments/StopGradientы
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 21
/layer_normalization_4/moments/SquaredDifferenceО
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indices
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2(
&layer_normalization_4/moments/variance
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752'
%layer_normalization_4/batchnorm/add/yъ
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2%
#layer_normalization_4/batchnorm/addЖ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2'
%layer_normalization_4/batchnorm/Rsqrtр
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpю
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_4/batchnorm/mulН
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/mul_1с
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/mul_2д
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpъ
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_4/batchnorm/subс
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/add_1е
-sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_2/dense_8/Tensordot/ReadVariableOp
#sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_8/Tensordot/axes
#sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_8/Tensordot/freeЅ
$sequential_2/dense_8/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/Shape
,sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/GatherV2/axisК
'sequential_2/dense_8/Tensordot/GatherV2GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/free:output:05sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/GatherV2Ђ
.sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_8/Tensordot/GatherV2_1/axisР
)sequential_2/dense_8/Tensordot/GatherV2_1GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/axes:output:07sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_8/Tensordot/GatherV2_1
$sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_8/Tensordot/Constд
#sequential_2/dense_8/Tensordot/ProdProd0sequential_2/dense_8/Tensordot/GatherV2:output:0-sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_8/Tensordot/Prod
&sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_1м
%sequential_2/dense_8/Tensordot/Prod_1Prod2sequential_2/dense_8/Tensordot/GatherV2_1:output:0/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_8/Tensordot/Prod_1
*sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_8/Tensordot/concat/axis
%sequential_2/dense_8/Tensordot/concatConcatV2,sequential_2/dense_8/Tensordot/free:output:0,sequential_2/dense_8/Tensordot/axes:output:03sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_8/Tensordot/concatр
$sequential_2/dense_8/Tensordot/stackPack,sequential_2/dense_8/Tensordot/Prod:output:0.sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/stackђ
(sequential_2/dense_8/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0.sequential_2/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2*
(sequential_2/dense_8/Tensordot/transposeѓ
&sequential_2/dense_8/Tensordot/ReshapeReshape,sequential_2/dense_8/Tensordot/transpose:y:0-sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2(
&sequential_2/dense_8/Tensordot/Reshapeђ
%sequential_2/dense_8/Tensordot/MatMulMatMul/sequential_2/dense_8/Tensordot/Reshape:output:05sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_2/dense_8/Tensordot/MatMul
&sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_2
,sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/concat_1/axisІ
'sequential_2/dense_8/Tensordot/concat_1ConcatV20sequential_2/dense_8/Tensordot/GatherV2:output:0/sequential_2/dense_8/Tensordot/Const_2:output:05sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/concat_1ф
sequential_2/dense_8/TensordotReshape/sequential_2/dense_8/Tensordot/MatMul:product:00sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
sequential_2/dense_8/TensordotЫ
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOpл
sequential_2/dense_8/BiasAddBiasAdd'sequential_2/dense_8/Tensordot:output:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_8/BiasAdd
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_8/Reluе
-sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_2/dense_9/Tensordot/ReadVariableOp
#sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_9/Tensordot/axes
#sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_9/Tensordot/freeЃ
$sequential_2/dense_9/Tensordot/ShapeShape'sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/Shape
,sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/GatherV2/axisК
'sequential_2/dense_9/Tensordot/GatherV2GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/free:output:05sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/GatherV2Ђ
.sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_9/Tensordot/GatherV2_1/axisР
)sequential_2/dense_9/Tensordot/GatherV2_1GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/axes:output:07sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_9/Tensordot/GatherV2_1
$sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_9/Tensordot/Constд
#sequential_2/dense_9/Tensordot/ProdProd0sequential_2/dense_9/Tensordot/GatherV2:output:0-sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_9/Tensordot/Prod
&sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_1м
%sequential_2/dense_9/Tensordot/Prod_1Prod2sequential_2/dense_9/Tensordot/GatherV2_1:output:0/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_9/Tensordot/Prod_1
*sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_9/Tensordot/concat/axis
%sequential_2/dense_9/Tensordot/concatConcatV2,sequential_2/dense_9/Tensordot/free:output:0,sequential_2/dense_9/Tensordot/axes:output:03sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_9/Tensordot/concatр
$sequential_2/dense_9/Tensordot/stackPack,sequential_2/dense_9/Tensordot/Prod:output:0.sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/stack№
(sequential_2/dense_9/Tensordot/transpose	Transpose'sequential_2/dense_8/Relu:activations:0.sequential_2/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2*
(sequential_2/dense_9/Tensordot/transposeѓ
&sequential_2/dense_9/Tensordot/ReshapeReshape,sequential_2/dense_9/Tensordot/transpose:y:0-sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2(
&sequential_2/dense_9/Tensordot/Reshapeђ
%sequential_2/dense_9/Tensordot/MatMulMatMul/sequential_2/dense_9/Tensordot/Reshape:output:05sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_2/dense_9/Tensordot/MatMul
&sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_2
,sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/concat_1/axisІ
'sequential_2/dense_9/Tensordot/concat_1ConcatV20sequential_2/dense_9/Tensordot/GatherV2:output:0/sequential_2/dense_9/Tensordot/Const_2:output:05sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/concat_1ф
sequential_2/dense_9/TensordotReshape/sequential_2/dense_9/Tensordot/MatMul:product:00sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
sequential_2/dense_9/TensordotЫ
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOpл
sequential_2/dense_9/BiasAddBiasAdd'sequential_2/dense_9/Tensordot:output:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_9/BiasAddw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout_9/dropout/ConstД
dropout_9/dropout/MulMul%sequential_2/dense_9/BiasAdd:output:0 dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_9/dropout/Mul
dropout_9/dropout/ShapeShape%sequential_2/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shapeж
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< *
dtype020
.dropout_9/dropout/random_uniform/RandomUniform
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2"
 dropout_9/dropout/GreaterEqual/yъ
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
dropout_9/dropout/GreaterEqualЁ
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ< 2
dropout_9/dropout/CastІ
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_9/dropout/Mul_1
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
add_1Ж
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indicesс
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2$
"layer_normalization_5/moments/meanЫ
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2,
*layer_normalization_5/moments/StopGradientэ
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 21
/layer_normalization_5/moments/SquaredDifferenceО
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indices
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2(
&layer_normalization_5/moments/variance
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752'
%layer_normalization_5/batchnorm/add/yъ
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2%
#layer_normalization_5/batchnorm/addЖ
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2'
%layer_normalization_5/batchnorm/Rsqrtр
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOpю
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_5/batchnorm/mulП
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/mul_1с
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/mul_2д
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOpъ
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_5/batchnorm/subс
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/add_1г
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp;^multi_head_attention_2/attention_output/add/ReadVariableOpE^multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_2/key/add/ReadVariableOp8^multi_head_attention_2/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/query/add/ReadVariableOp:^multi_head_attention_2/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/value/add/ReadVariableOp:^multi_head_attention_2/value/einsum/Einsum/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp.^sequential_2/dense_8/Tensordot/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp.^sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:џџџџџџџџџ< ::::::::::::::::2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_2/attention_output/add/ReadVariableOp:multi_head_attention_2/attention_output/add/ReadVariableOp2
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_2/key/add/ReadVariableOp-multi_head_attention_2/key/add/ReadVariableOp2r
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/query/add/ReadVariableOp/multi_head_attention_2/query/add/ReadVariableOp2v
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/value/add/ReadVariableOp/multi_head_attention_2/value/add/ReadVariableOp2v
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2^
-sequential_2/dense_8/Tensordot/ReadVariableOp-sequential_2/dense_8/Tensordot/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2^
-sequential_2/dense_9/Tensordot/ReadVariableOp-sequential_2/dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs

F
*__inference_dropout_10_layer_call_fn_10147

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_88322
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

c
D__inference_dropout_10_layer_call_and_return_conditional_losses_8827

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ё
V
:__inference_global_average_pooling1d_2_layer_call_fn_10109

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_83722
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А)

A__inference_model_2_layer_call_and_return_conditional_losses_9147

inputs'
#token_and_position_embedding_2_9095'
#token_and_position_embedding_2_9097
transformer_block_2_9100
transformer_block_2_9102
transformer_block_2_9104
transformer_block_2_9106
transformer_block_2_9108
transformer_block_2_9110
transformer_block_2_9112
transformer_block_2_9114
transformer_block_2_9116
transformer_block_2_9118
transformer_block_2_9120
transformer_block_2_9122
transformer_block_2_9124
transformer_block_2_9126
transformer_block_2_9128
transformer_block_2_9130
dense_10_9135
dense_10_9137
dense_11_9141
dense_11_9143
identityЂ dense_10/StatefulPartitionedCallЂ dense_11/StatefulPartitionedCallЂ6token_and_position_embedding_2/StatefulPartitionedCallЂ+transformer_block_2/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs#token_and_position_embedding_2_9095#token_and_position_embedding_2_9097*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_840328
6token_and_position_embedding_2/StatefulPartitionedCall
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_9100transformer_block_2_9102transformer_block_2_9104transformer_block_2_9106transformer_block_2_9108transformer_block_2_9110transformer_block_2_9112transformer_block_2_9114transformer_block_2_9116transformer_block_2_9118transformer_block_2_9120transformer_block_2_9122transformer_block_2_9124transformer_block_2_9126transformer_block_2_9128transformer_block_2_9130*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_86942-
+transformer_block_2/StatefulPartitionedCallЖ
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_88082,
*global_average_pooling1d_2/PartitionedCall
dropout_10/PartitionedCallPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_88322
dropout_10/PartitionedCallЋ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_10_9135dense_10_9137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_88562"
 dense_10/StatefulPartitionedCallћ
dropout_11/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_88892
dropout_11/PartitionedCallЋ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_11_9141dense_11_9143*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_89132"
 dense_11/StatefulPartitionedCallЊ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs

d
E__inference_dropout_10_layer_call_and_return_conditional_losses_10132

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

c
D__inference_dropout_11_layer_call_and_return_conditional_losses_8884

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ
ј
F__inference_sequential_2_layer_call_and_return_conditional_losses_8287
dense_8_input
dense_8_8235
dense_8_8237
dense_9_8281
dense_9_8283
identityЂdense_8/StatefulPartitionedCallЂdense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_8235dense_8_8237*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_82242!
dense_8/StatefulPartitionedCallЏ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_8281dense_9_8283*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_82702!
dense_9/StatefulPartitionedCallФ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ< ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ< 
'
_user_specified_namedense_8_input
л
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_10115

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ< :S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
№
Г
&__inference_model_2_layer_call_fn_9194
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_91472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ<
!
_user_specified_name	input_4
э
В
&__inference_model_2_layer_call_fn_9716

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_91472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
м
б
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_10024

inputsF
Bmulti_head_attention_2_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_query_add_readvariableop_resourceD
@multi_head_attention_2_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_2_key_add_readvariableop_resourceF
Bmulti_head_attention_2_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_value_add_readvariableop_resourceQ
Mmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_2_attention_output_add_readvariableop_resource?
;layer_normalization_4_batchnorm_mul_readvariableop_resource;
7layer_normalization_4_batchnorm_readvariableop_resource:
6sequential_2_dense_8_tensordot_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resource:
6sequential_2_dense_9_tensordot_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resource?
;layer_normalization_5_batchnorm_mul_readvariableop_resource;
7layer_normalization_5_batchnorm_readvariableop_resource
identityЂ.layer_normalization_4/batchnorm/ReadVariableOpЂ2layer_normalization_4/batchnorm/mul/ReadVariableOpЂ.layer_normalization_5/batchnorm/ReadVariableOpЂ2layer_normalization_5/batchnorm/mul/ReadVariableOpЂ:multi_head_attention_2/attention_output/add/ReadVariableOpЂDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpЂ-multi_head_attention_2/key/add/ReadVariableOpЂ7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpЂ/multi_head_attention_2/query/add/ReadVariableOpЂ9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpЂ/multi_head_attention_2/value/add/ReadVariableOpЂ9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpЂ+sequential_2/dense_8/BiasAdd/ReadVariableOpЂ-sequential_2/dense_8/Tensordot/ReadVariableOpЂ+sequential_2/dense_9/BiasAdd/ReadVariableOpЂ-sequential_2/dense_9/Tensordot/ReadVariableOp§
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_2/query/einsum/EinsumEinsuminputsAmulti_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2,
*multi_head_attention_2/query/einsum/Einsumл
/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp8multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_2/query/add/ReadVariableOpѕ
 multi_head_attention_2/query/addAddV23multi_head_attention_2/query/einsum/Einsum:output:07multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2"
 multi_head_attention_2/query/addї
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_2/key/einsum/EinsumEinsuminputs?multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2*
(multi_head_attention_2/key/einsum/Einsumе
-multi_head_attention_2/key/add/ReadVariableOpReadVariableOp6multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_2/key/add/ReadVariableOpэ
multi_head_attention_2/key/addAddV21multi_head_attention_2/key/einsum/Einsum:output:05multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2 
multi_head_attention_2/key/add§
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_2/value/einsum/EinsumEinsuminputsAmulti_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2,
*multi_head_attention_2/value/einsum/Einsumл
/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp8multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_2/value/add/ReadVariableOpѕ
 multi_head_attention_2/value/addAddV23multi_head_attention_2/value/einsum/Einsum:output:07multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2"
 multi_head_attention_2/value/add
multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓ5>2
multi_head_attention_2/Mul/yЦ
multi_head_attention_2/MulMul$multi_head_attention_2/query/add:z:0%multi_head_attention_2/Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2
multi_head_attention_2/Mulќ
$multi_head_attention_2/einsum/EinsumEinsum"multi_head_attention_2/key/add:z:0multi_head_attention_2/Mul:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ<<*
equationaecd,abcd->acbe2&
$multi_head_attention_2/einsum/EinsumФ
&multi_head_attention_2/softmax/SoftmaxSoftmax-multi_head_attention_2/einsum/Einsum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2(
&multi_head_attention_2/softmax/SoftmaxЪ
'multi_head_attention_2/dropout/IdentityIdentity0multi_head_attention_2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2)
'multi_head_attention_2/dropout/Identity
&multi_head_attention_2/einsum_1/EinsumEinsum0multi_head_attention_2/dropout/Identity:output:0$multi_head_attention_2/value/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationacbe,aecd->abcd2(
&multi_head_attention_2/einsum_1/Einsum
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpг
5multi_head_attention_2/attention_output/einsum/EinsumEinsum/multi_head_attention_2/einsum_1/Einsum:output:0Lmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ< *
equationabcd,cde->abe27
5multi_head_attention_2/attention_output/einsum/Einsumј
:multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_2/attention_output/add/ReadVariableOp
+multi_head_attention_2/attention_output/addAddV2>multi_head_attention_2/attention_output/einsum/Einsum:output:0Bmulti_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2-
+multi_head_attention_2/attention_output/add
dropout_8/IdentityIdentity/multi_head_attention_2/attention_output/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_8/Identityn
addAddV2inputsdropout_8/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
addЖ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesп
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2$
"layer_normalization_4/moments/meanЫ
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2,
*layer_normalization_4/moments/StopGradientы
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 21
/layer_normalization_4/moments/SquaredDifferenceО
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indices
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2(
&layer_normalization_4/moments/variance
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752'
%layer_normalization_4/batchnorm/add/yъ
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2%
#layer_normalization_4/batchnorm/addЖ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2'
%layer_normalization_4/batchnorm/Rsqrtр
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpю
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_4/batchnorm/mulН
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/mul_1с
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/mul_2д
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpъ
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_4/batchnorm/subс
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_4/batchnorm/add_1е
-sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_2/dense_8/Tensordot/ReadVariableOp
#sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_8/Tensordot/axes
#sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_8/Tensordot/freeЅ
$sequential_2/dense_8/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/Shape
,sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/GatherV2/axisК
'sequential_2/dense_8/Tensordot/GatherV2GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/free:output:05sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/GatherV2Ђ
.sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_8/Tensordot/GatherV2_1/axisР
)sequential_2/dense_8/Tensordot/GatherV2_1GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/axes:output:07sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_8/Tensordot/GatherV2_1
$sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_8/Tensordot/Constд
#sequential_2/dense_8/Tensordot/ProdProd0sequential_2/dense_8/Tensordot/GatherV2:output:0-sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_8/Tensordot/Prod
&sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_1м
%sequential_2/dense_8/Tensordot/Prod_1Prod2sequential_2/dense_8/Tensordot/GatherV2_1:output:0/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_8/Tensordot/Prod_1
*sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_8/Tensordot/concat/axis
%sequential_2/dense_8/Tensordot/concatConcatV2,sequential_2/dense_8/Tensordot/free:output:0,sequential_2/dense_8/Tensordot/axes:output:03sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_8/Tensordot/concatр
$sequential_2/dense_8/Tensordot/stackPack,sequential_2/dense_8/Tensordot/Prod:output:0.sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/stackђ
(sequential_2/dense_8/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0.sequential_2/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2*
(sequential_2/dense_8/Tensordot/transposeѓ
&sequential_2/dense_8/Tensordot/ReshapeReshape,sequential_2/dense_8/Tensordot/transpose:y:0-sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2(
&sequential_2/dense_8/Tensordot/Reshapeђ
%sequential_2/dense_8/Tensordot/MatMulMatMul/sequential_2/dense_8/Tensordot/Reshape:output:05sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_2/dense_8/Tensordot/MatMul
&sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_2
,sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/concat_1/axisІ
'sequential_2/dense_8/Tensordot/concat_1ConcatV20sequential_2/dense_8/Tensordot/GatherV2:output:0/sequential_2/dense_8/Tensordot/Const_2:output:05sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/concat_1ф
sequential_2/dense_8/TensordotReshape/sequential_2/dense_8/Tensordot/MatMul:product:00sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
sequential_2/dense_8/TensordotЫ
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOpл
sequential_2/dense_8/BiasAddBiasAdd'sequential_2/dense_8/Tensordot:output:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_8/BiasAdd
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_8/Reluе
-sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_2/dense_9/Tensordot/ReadVariableOp
#sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_9/Tensordot/axes
#sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_9/Tensordot/freeЃ
$sequential_2/dense_9/Tensordot/ShapeShape'sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/Shape
,sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/GatherV2/axisК
'sequential_2/dense_9/Tensordot/GatherV2GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/free:output:05sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/GatherV2Ђ
.sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_9/Tensordot/GatherV2_1/axisР
)sequential_2/dense_9/Tensordot/GatherV2_1GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/axes:output:07sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_9/Tensordot/GatherV2_1
$sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_9/Tensordot/Constд
#sequential_2/dense_9/Tensordot/ProdProd0sequential_2/dense_9/Tensordot/GatherV2:output:0-sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_9/Tensordot/Prod
&sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_1м
%sequential_2/dense_9/Tensordot/Prod_1Prod2sequential_2/dense_9/Tensordot/GatherV2_1:output:0/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_9/Tensordot/Prod_1
*sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_9/Tensordot/concat/axis
%sequential_2/dense_9/Tensordot/concatConcatV2,sequential_2/dense_9/Tensordot/free:output:0,sequential_2/dense_9/Tensordot/axes:output:03sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_9/Tensordot/concatр
$sequential_2/dense_9/Tensordot/stackPack,sequential_2/dense_9/Tensordot/Prod:output:0.sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/stack№
(sequential_2/dense_9/Tensordot/transpose	Transpose'sequential_2/dense_8/Relu:activations:0.sequential_2/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2*
(sequential_2/dense_9/Tensordot/transposeѓ
&sequential_2/dense_9/Tensordot/ReshapeReshape,sequential_2/dense_9/Tensordot/transpose:y:0-sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2(
&sequential_2/dense_9/Tensordot/Reshapeђ
%sequential_2/dense_9/Tensordot/MatMulMatMul/sequential_2/dense_9/Tensordot/Reshape:output:05sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_2/dense_9/Tensordot/MatMul
&sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_2
,sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/concat_1/axisІ
'sequential_2/dense_9/Tensordot/concat_1ConcatV20sequential_2/dense_9/Tensordot/GatherV2:output:0/sequential_2/dense_9/Tensordot/Const_2:output:05sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/concat_1ф
sequential_2/dense_9/TensordotReshape/sequential_2/dense_9/Tensordot/MatMul:product:00sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2 
sequential_2/dense_9/TensordotЫ
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOpл
sequential_2/dense_9/BiasAddBiasAdd'sequential_2/dense_9/Tensordot:output:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
sequential_2/dense_9/BiasAdd
dropout_9/IdentityIdentity%sequential_2/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
dropout_9/Identity
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_9/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
add_1Ж
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indicesс
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2$
"layer_normalization_5/moments/meanЫ
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2,
*layer_normalization_5/moments/StopGradientэ
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 21
/layer_normalization_5/moments/SquaredDifferenceО
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indices
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2(
&layer_normalization_5/moments/variance
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752'
%layer_normalization_5/batchnorm/add/yъ
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2%
#layer_normalization_5/batchnorm/addЖ
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2'
%layer_normalization_5/batchnorm/Rsqrtр
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOpю
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_5/batchnorm/mulП
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/mul_1с
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/mul_2д
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOpъ
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2%
#layer_normalization_5/batchnorm/subс
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2'
%layer_normalization_5/batchnorm/add_1г
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp;^multi_head_attention_2/attention_output/add/ReadVariableOpE^multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_2/key/add/ReadVariableOp8^multi_head_attention_2/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/query/add/ReadVariableOp:^multi_head_attention_2/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/value/add/ReadVariableOp:^multi_head_attention_2/value/einsum/Einsum/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp.^sequential_2/dense_8/Tensordot/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp.^sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:џџџџџџџџџ< ::::::::::::::::2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_2/attention_output/add/ReadVariableOp:multi_head_attention_2/attention_output/add/ReadVariableOp2
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_2/key/add/ReadVariableOp-multi_head_attention_2/key/add/ReadVariableOp2r
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/query/add/ReadVariableOp/multi_head_attention_2/query/add/ReadVariableOp2v
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/value/add/ReadVariableOp/multi_head_attention_2/value/add/ReadVariableOp2v
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2^
-sequential_2/dense_8/Tensordot/ReadVariableOp-sequential_2/dense_8/Tensordot/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2^
-sequential_2/dense_9/Tensordot/ReadVariableOp-sequential_2/dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs

Ч
__inference__wrapped_model_8189
input_4L
Hmodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_8035L
Hmodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_8041b
^model_2_transformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resourceX
Tmodel_2_transformer_block_2_multi_head_attention_2_query_add_readvariableop_resource`
\model_2_transformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resourceV
Rmodel_2_transformer_block_2_multi_head_attention_2_key_add_readvariableop_resourceb
^model_2_transformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resourceX
Tmodel_2_transformer_block_2_multi_head_attention_2_value_add_readvariableop_resourcem
imodel_2_transformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resourcec
_model_2_transformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resource[
Wmodel_2_transformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resourceW
Smodel_2_transformer_block_2_layer_normalization_4_batchnorm_readvariableop_resourceV
Rmodel_2_transformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resourceT
Pmodel_2_transformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resourceV
Rmodel_2_transformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resourceT
Pmodel_2_transformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resource[
Wmodel_2_transformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resourceW
Smodel_2_transformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource3
/model_2_dense_10_matmul_readvariableop_resource4
0model_2_dense_10_biasadd_readvariableop_resource3
/model_2_dense_11_matmul_readvariableop_resource4
0model_2_dense_11_biasadd_readvariableop_resource
identityЂ'model_2/dense_10/BiasAdd/ReadVariableOpЂ&model_2/dense_10/MatMul/ReadVariableOpЂ'model_2/dense_11/BiasAdd/ReadVariableOpЂ&model_2/dense_11/MatMul/ReadVariableOpЂCmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupЂCmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupЂJmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpЂNmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpЂJmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpЂNmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpЂVmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpЂ`model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpЂImodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpЂSmodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpЂKmodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpЂUmodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpЂKmodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpЂUmodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpЂGmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpЂImodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpЂGmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpЂImodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp
,model_2/token_and_position_embedding_2/ShapeShapeinput_4*
T0*
_output_shapes
:2.
,model_2/token_and_position_embedding_2/ShapeЫ
:model_2/token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2<
:model_2/token_and_position_embedding_2/strided_slice/stackЦ
<model_2/token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_2/token_and_position_embedding_2/strided_slice/stack_1Ц
<model_2/token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_2/token_and_position_embedding_2/strided_slice/stack_2Ь
4model_2/token_and_position_embedding_2/strided_sliceStridedSlice5model_2/token_and_position_embedding_2/Shape:output:0Cmodel_2/token_and_position_embedding_2/strided_slice/stack:output:0Emodel_2/token_and_position_embedding_2/strided_slice/stack_1:output:0Emodel_2/token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_2/token_and_position_embedding_2/strided_sliceЊ
2model_2/token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_2/token_and_position_embedding_2/range/startЊ
2model_2/token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_2/token_and_position_embedding_2/range/deltaУ
,model_2/token_and_position_embedding_2/rangeRange;model_2/token_and_position_embedding_2/range/start:output:0=model_2/token_and_position_embedding_2/strided_slice:output:0;model_2/token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:џџџџџџџџџ2.
,model_2/token_and_position_embedding_2/rangeю
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherHmodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_80355model_2/token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*[
_classQ
OMloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/8035*'
_output_shapes
:џџџџџџџџџ *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupГ
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*[
_classQ
OMloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/8035*'
_output_shapes
:џџџџџџџџџ 2N
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/IdentityЕ
Nmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2P
Nmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1Ф
7model_2/token_and_position_embedding_2/embedding_4/CastCastinput_4*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ<29
7model_2/token_and_position_embedding_2/embedding_4/Castј
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherHmodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_8041;model_2/token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*[
_classQ
OMloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/8041*+
_output_shapes
:џџџџџџџџџ< *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupЗ
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*[
_classQ
OMloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/8041*+
_output_shapes
:џџџџџџџџџ< 2N
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/IdentityЙ
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2P
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1Щ
*model_2/token_and_position_embedding_2/addAddV2Wmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Wmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2,
*model_2/token_and_position_embedding_2/addб
Umodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp
Fmodel_2/transformer_block_2/multi_head_attention_2/query/einsum/EinsumEinsum.model_2/token_and_position_embedding_2/add:z:0]model_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2H
Fmodel_2/transformer_block_2/multi_head_attention_2/query/einsum/EinsumЏ
Kmodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_2_multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpх
<model_2/transformer_block_2/multi_head_attention_2/query/addAddV2Omodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum:output:0Smodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2>
<model_2/transformer_block_2/multi_head_attention_2/query/addЫ
Smodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_2_transformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp
Dmodel_2/transformer_block_2/multi_head_attention_2/key/einsum/EinsumEinsum.model_2/token_and_position_embedding_2/add:z:0[model_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2F
Dmodel_2/transformer_block_2/multi_head_attention_2/key/einsum/EinsumЉ
Imodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpReadVariableOpRmodel_2_transformer_block_2_multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpн
:model_2/transformer_block_2/multi_head_attention_2/key/addAddV2Mmodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum:output:0Qmodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2<
:model_2/transformer_block_2/multi_head_attention_2/key/addб
Umodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp
Fmodel_2/transformer_block_2/multi_head_attention_2/value/einsum/EinsumEinsum.model_2/token_and_position_embedding_2/add:z:0]model_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationabc,cde->abde2H
Fmodel_2/transformer_block_2/multi_head_attention_2/value/einsum/EinsumЏ
Kmodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_2_multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpх
<model_2/transformer_block_2/multi_head_attention_2/value/addAddV2Omodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum:output:0Smodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ< 2>
<model_2/transformer_block_2/multi_head_attention_2/value/addЙ
8model_2/transformer_block_2/multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓ5>2:
8model_2/transformer_block_2/multi_head_attention_2/Mul/yЖ
6model_2/transformer_block_2/multi_head_attention_2/MulMul@model_2/transformer_block_2/multi_head_attention_2/query/add:z:0Amodel_2/transformer_block_2/multi_head_attention_2/Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ< 28
6model_2/transformer_block_2/multi_head_attention_2/Mulь
@model_2/transformer_block_2/multi_head_attention_2/einsum/EinsumEinsum>model_2/transformer_block_2/multi_head_attention_2/key/add:z:0:model_2/transformer_block_2/multi_head_attention_2/Mul:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ<<*
equationaecd,abcd->acbe2B
@model_2/transformer_block_2/multi_head_attention_2/einsum/Einsum
Bmodel_2/transformer_block_2/multi_head_attention_2/softmax/SoftmaxSoftmaxImodel_2/transformer_block_2/multi_head_attention_2/einsum/Einsum:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2D
Bmodel_2/transformer_block_2/multi_head_attention_2/softmax/Softmax
Cmodel_2/transformer_block_2/multi_head_attention_2/dropout/IdentityIdentityLmodel_2/transformer_block_2/multi_head_attention_2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџ<<2E
Cmodel_2/transformer_block_2/multi_head_attention_2/dropout/Identity
Bmodel_2/transformer_block_2/multi_head_attention_2/einsum_1/EinsumEinsumLmodel_2/transformer_block_2/multi_head_attention_2/dropout/Identity:output:0@model_2/transformer_block_2/multi_head_attention_2/value/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ< *
equationacbe,aecd->abcd2D
Bmodel_2/transformer_block_2/multi_head_attention_2/einsum_1/Einsumђ
`model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_2_transformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02b
`model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpУ
Qmodel_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/EinsumEinsumKmodel_2/transformer_block_2/multi_head_attention_2/einsum_1/Einsum:output:0hmodel_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ< *
equationabcd,cde->abe2S
Qmodel_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/EinsumЬ
Vmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOp_model_2_transformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp
Gmodel_2/transformer_block_2/multi_head_attention_2/attention_output/addAddV2Zmodel_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum:output:0^model_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2I
Gmodel_2/transformer_block_2/multi_head_attention_2/attention_output/addя
.model_2/transformer_block_2/dropout_8/IdentityIdentityKmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 20
.model_2/transformer_block_2/dropout_8/Identityъ
model_2/transformer_block_2/addAddV2.model_2/token_and_position_embedding_2/add:z:07model_2/transformer_block_2/dropout_8/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2!
model_2/transformer_block_2/addю
Pmodel_2/transformer_block_2/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_2/transformer_block_2/layer_normalization_4/moments/mean/reduction_indicesЯ
>model_2/transformer_block_2/layer_normalization_4/moments/meanMean#model_2/transformer_block_2/add:z:0Ymodel_2/transformer_block_2/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2@
>model_2/transformer_block_2/layer_normalization_4/moments/mean
Fmodel_2/transformer_block_2/layer_normalization_4/moments/StopGradientStopGradientGmodel_2/transformer_block_2/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2H
Fmodel_2/transformer_block_2/layer_normalization_4/moments/StopGradientл
Kmodel_2/transformer_block_2/layer_normalization_4/moments/SquaredDifferenceSquaredDifference#model_2/transformer_block_2/add:z:0Omodel_2/transformer_block_2/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2M
Kmodel_2/transformer_block_2/layer_normalization_4/moments/SquaredDifferenceі
Tmodel_2/transformer_block_2/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2V
Tmodel_2/transformer_block_2/layer_normalization_4/moments/variance/reduction_indices
Bmodel_2/transformer_block_2/layer_normalization_4/moments/varianceMeanOmodel_2/transformer_block_2/layer_normalization_4/moments/SquaredDifference:z:0]model_2/transformer_block_2/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2D
Bmodel_2/transformer_block_2/layer_normalization_4/moments/varianceЫ
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add/yк
?model_2/transformer_block_2/layer_normalization_4/batchnorm/addAddV2Kmodel_2/transformer_block_2/layer_normalization_4/moments/variance:output:0Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2A
?model_2/transformer_block_2/layer_normalization_4/batchnorm/add
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/RsqrtRsqrtCmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/RsqrtД
Nmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_2_transformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02P
Nmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpо
?model_2/transformer_block_2/layer_normalization_4/batchnorm/mulMulEmodel_2/transformer_block_2/layer_normalization_4/batchnorm/Rsqrt:y:0Vmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2A
?model_2/transformer_block_2/layer_normalization_4/batchnorm/mul­
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_1Mul#model_2/transformer_block_2/add:z:0Cmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_1б
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_2MulGmodel_2/transformer_block_2/layer_normalization_4/moments/mean:output:0Cmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_2Ј
Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpSmodel_2_transformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpк
?model_2/transformer_block_2/layer_normalization_4/batchnorm/subSubRmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp:value:0Emodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2A
?model_2/transformer_block_2/layer_normalization_4/batchnorm/subб
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1AddV2Emodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_1:z:0Cmodel_2/transformer_block_2/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1Љ
Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOpRmodel_2_transformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02K
Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpЬ
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2A
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/axesг
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/freeљ
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/ShapeShapeEmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2B
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/Shapeж
Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axisЦ
Cmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2GatherV2Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0Qmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2к
Jmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axisЬ
Emodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1GatherV2Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Smodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2G
Emodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1Ю
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2B
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/ConstФ
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/ProdProdLmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2A
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/Prodв
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_1Ь
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1ProdNmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1:output:0Kmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2C
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1в
Fmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat/axisЅ
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concatConcatV2Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Omodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concatа
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/stackPackHmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Prod:output:0Jmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2B
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/stackт
Dmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/transpose	TransposeEmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0Jmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2F
Dmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/transposeу
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReshapeReshapeHmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/transpose:y:0Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2D
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Reshapeт
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/MatMulMatMulKmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Reshape:output:0Qmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2C
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/MatMulв
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2D
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_2ж
Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axisВ
Cmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1ConcatV2Lmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Kmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_2:output:0Qmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2E
Cmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1д
:model_2/transformer_block_2/sequential_2/dense_8/TensordotReshapeKmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/MatMul:product:0Lmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2<
:model_2/transformer_block_2/sequential_2/dense_8/Tensordot
Gmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOpPmodel_2_transformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02I
Gmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpЫ
8model_2/transformer_block_2/sequential_2/dense_8/BiasAddBiasAddCmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot:output:0Omodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2:
8model_2/transformer_block_2/sequential_2/dense_8/BiasAddя
5model_2/transformer_block_2/sequential_2/dense_8/ReluReluAmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 27
5model_2/transformer_block_2/sequential_2/dense_8/ReluЉ
Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOpRmodel_2_transformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02K
Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpЬ
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2A
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/axesг
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/freeї
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/ShapeShapeCmodel_2/transformer_block_2/sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2B
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/Shapeж
Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axisЦ
Cmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2GatherV2Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0Qmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2к
Jmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axisЬ
Emodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1GatherV2Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Smodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2G
Emodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1Ю
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2B
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/ConstФ
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/ProdProdLmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2A
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/Prodв
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_1Ь
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1ProdNmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1:output:0Kmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2C
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1в
Fmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat/axisЅ
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concatConcatV2Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Omodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concatа
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/stackPackHmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Prod:output:0Jmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2B
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/stackр
Dmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/transpose	TransposeCmodel_2/transformer_block_2/sequential_2/dense_8/Relu:activations:0Jmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2F
Dmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/transposeу
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReshapeReshapeHmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/transpose:y:0Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2D
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Reshapeт
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/MatMulMatMulKmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Reshape:output:0Qmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2C
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/MatMulв
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2D
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_2ж
Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axisВ
Cmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1ConcatV2Lmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Kmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_2:output:0Qmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2E
Cmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1д
:model_2/transformer_block_2/sequential_2/dense_9/TensordotReshapeKmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/MatMul:product:0Lmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2<
:model_2/transformer_block_2/sequential_2/dense_9/Tensordot
Gmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOpPmodel_2_transformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02I
Gmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpЫ
8model_2/transformer_block_2/sequential_2/dense_9/BiasAddBiasAddCmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot:output:0Omodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2:
8model_2/transformer_block_2/sequential_2/dense_9/BiasAddх
.model_2/transformer_block_2/dropout_9/IdentityIdentityAmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 20
.model_2/transformer_block_2/dropout_9/Identity
!model_2/transformer_block_2/add_1AddV2Emodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1:z:07model_2/transformer_block_2/dropout_9/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2#
!model_2/transformer_block_2/add_1ю
Pmodel_2/transformer_block_2/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_2/transformer_block_2/layer_normalization_5/moments/mean/reduction_indicesб
>model_2/transformer_block_2/layer_normalization_5/moments/meanMean%model_2/transformer_block_2/add_1:z:0Ymodel_2/transformer_block_2/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2@
>model_2/transformer_block_2/layer_normalization_5/moments/mean
Fmodel_2/transformer_block_2/layer_normalization_5/moments/StopGradientStopGradientGmodel_2/transformer_block_2/layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2H
Fmodel_2/transformer_block_2/layer_normalization_5/moments/StopGradientн
Kmodel_2/transformer_block_2/layer_normalization_5/moments/SquaredDifferenceSquaredDifference%model_2/transformer_block_2/add_1:z:0Omodel_2/transformer_block_2/layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2M
Kmodel_2/transformer_block_2/layer_normalization_5/moments/SquaredDifferenceі
Tmodel_2/transformer_block_2/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2V
Tmodel_2/transformer_block_2/layer_normalization_5/moments/variance/reduction_indices
Bmodel_2/transformer_block_2/layer_normalization_5/moments/varianceMeanOmodel_2/transformer_block_2/layer_normalization_5/moments/SquaredDifference:z:0]model_2/transformer_block_2/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<*
	keep_dims(2D
Bmodel_2/transformer_block_2/layer_normalization_5/moments/varianceЫ
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н752C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add/yк
?model_2/transformer_block_2/layer_normalization_5/batchnorm/addAddV2Kmodel_2/transformer_block_2/layer_normalization_5/moments/variance:output:0Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2A
?model_2/transformer_block_2/layer_normalization_5/batchnorm/add
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/RsqrtRsqrtCmodel_2/transformer_block_2/layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ<2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/RsqrtД
Nmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_2_transformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02P
Nmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpо
?model_2/transformer_block_2/layer_normalization_5/batchnorm/mulMulEmodel_2/transformer_block_2/layer_normalization_5/batchnorm/Rsqrt:y:0Vmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2A
?model_2/transformer_block_2/layer_normalization_5/batchnorm/mulЏ
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_1Mul%model_2/transformer_block_2/add_1:z:0Cmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_1б
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_2MulGmodel_2/transformer_block_2/layer_normalization_5/moments/mean:output:0Cmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_2Ј
Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOpSmodel_2_transformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpк
?model_2/transformer_block_2/layer_normalization_5/batchnorm/subSubRmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp:value:0Emodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2A
?model_2/transformer_block_2/layer_normalization_5/batchnorm/subб
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add_1AddV2Emodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_1:z:0Cmodel_2/transformer_block_2/layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add_1И
9model_2/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9model_2/global_average_pooling1d_2/Mean/reduction_indices
'model_2/global_average_pooling1d_2/MeanMeanEmodel_2/transformer_block_2/layer_normalization_5/batchnorm/add_1:z:0Bmodel_2/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'model_2/global_average_pooling1d_2/MeanЊ
model_2/dropout_10/IdentityIdentity0model_2/global_average_pooling1d_2/Mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_2/dropout_10/IdentityР
&model_2/dense_10/MatMul/ReadVariableOpReadVariableOp/model_2_dense_10_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&model_2/dense_10/MatMul/ReadVariableOpФ
model_2/dense_10/MatMulMatMul$model_2/dropout_10/Identity:output:0.model_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_2/dense_10/MatMulП
'model_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_2/dense_10/BiasAdd/ReadVariableOpХ
model_2/dense_10/BiasAddBiasAdd!model_2/dense_10/MatMul:product:0/model_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_2/dense_10/BiasAdd
model_2/dense_10/ReluRelu!model_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_2/dense_10/Relu
model_2/dropout_11/IdentityIdentity#model_2/dense_10/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_2/dropout_11/IdentityР
&model_2/dense_11/MatMul/ReadVariableOpReadVariableOp/model_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_11/MatMul/ReadVariableOpФ
model_2/dense_11/MatMulMatMul$model_2/dropout_11/Identity:output:0.model_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_2/dense_11/MatMulП
'model_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_2/dense_11/BiasAdd/ReadVariableOpХ
model_2/dense_11/BiasAddBiasAdd!model_2/dense_11/MatMul:product:0/model_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_2/dense_11/BiasAdd
model_2/dense_11/SoftmaxSoftmax!model_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_2/dense_11/SoftmaxК
IdentityIdentity"model_2/dense_11/Softmax:softmax:0(^model_2/dense_10/BiasAdd/ReadVariableOp'^model_2/dense_10/MatMul/ReadVariableOp(^model_2/dense_11/BiasAdd/ReadVariableOp'^model_2/dense_11/MatMul/ReadVariableOpD^model_2/token_and_position_embedding_2/embedding_4/embedding_lookupD^model_2/token_and_position_embedding_2/embedding_5/embedding_lookupK^model_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpO^model_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpK^model_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpO^model_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpW^model_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpa^model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpJ^model_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpT^model_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpV^model_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpV^model_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpH^model_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpJ^model_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpH^model_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpJ^model_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::2R
'model_2/dense_10/BiasAdd/ReadVariableOp'model_2/dense_10/BiasAdd/ReadVariableOp2P
&model_2/dense_10/MatMul/ReadVariableOp&model_2/dense_10/MatMul/ReadVariableOp2R
'model_2/dense_11/BiasAdd/ReadVariableOp'model_2/dense_11/BiasAdd/ReadVariableOp2P
&model_2/dense_11/MatMul/ReadVariableOp&model_2/dense_11/MatMul/ReadVariableOp2
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupCmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup2
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupCmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup2
Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpJmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp2 
Nmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpNmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp2
Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpJmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp2 
Nmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpNmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp2А
Vmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpVmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp2Ф
`model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp`model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2
Imodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpImodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOp2Њ
Smodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpSmodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2
Kmodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpKmodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOp2Ў
Umodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpUmodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2
Kmodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpKmodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOp2Ў
Umodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpUmodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2
Gmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpGmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp2
Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpImodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp2
Gmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpGmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp2
Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpImodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџ<
!
_user_specified_name	input_4
э
В
&__inference_model_2_layer_call_fn_9667

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_90432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
Л
V
:__inference_global_average_pooling1d_2_layer_call_fn_10120

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_88082
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ< :S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
Ц
Ѕ
+__inference_sequential_2_layer_call_fn_8356
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_83452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ< ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ< 
'
_user_specified_namedense_8_input

p
T__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_8372

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_10137

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

d
E__inference_dropout_11_layer_call_and_return_conditional_losses_10179

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
ў
X__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_9740
x%
!embedding_5_embedding_lookup_9727%
!embedding_4_embedding_lookup_9733
identityЂembedding_4/embedding_lookupЂembedding_5/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:џџџџџџџџџ2
rangeЋ
embedding_5/embedding_lookupResourceGather!embedding_5_embedding_lookup_9727range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_5/embedding_lookup/9727*'
_output_shapes
:џџџџџџџџџ *
dtype02
embedding_5/embedding_lookup
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_5/embedding_lookup/9727*'
_output_shapes
:џџџџџџџџџ 2'
%embedding_5/embedding_lookup/IdentityР
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'embedding_5/embedding_lookup/Identity_1p
embedding_4/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ<2
embedding_4/CastЕ
embedding_4/embedding_lookupResourceGather!embedding_4_embedding_lookup_9733embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_4/embedding_lookup/9733*+
_output_shapes
:џџџџџџџџџ< *
dtype02
embedding_4/embedding_lookup
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_4/embedding_lookup/9733*+
_output_shapes
:џџџџџџџџџ< 2'
%embedding_4/embedding_lookup/IdentityФ
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2)
'embedding_4/embedding_lookup/Identity_1­
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
add
IdentityIdentityadd:z:0^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ<::2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:J F
'
_output_shapes
:џџџџџџџџџ<

_user_specified_namex
э	
м
C__inference_dense_10_layer_call_and_return_conditional_losses_10158

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
й
}
(__inference_dense_10_layer_call_fn_10167

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_88562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ў 
р
A__inference_dense_8_layer_call_and_return_conditional_losses_8224

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ< 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
В

,__inference_sequential_2_layer_call_fn_10341

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_83182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ< ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
к
p
T__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_8808

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ< :S O
+
_output_shapes
:џџџџџџџџџ< 
 
_user_specified_nameinputs
Ъ
Џ
"__inference_signature_wrapper_9253
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_81892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ<
!
_user_specified_name	input_4
Ч
b
D__inference_dropout_11_layer_call_and_return_conditional_losses_8889

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ
ј
F__inference_sequential_2_layer_call_and_return_conditional_losses_8301
dense_8_input
dense_8_8290
dense_8_8292
dense_9_8295
dense_9_8297
identityЂdense_8/StatefulPartitionedCallЂdense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_8290dense_8_8292*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_82242!
dense_8/StatefulPartitionedCallЏ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_8295dense_9_8297*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_82702!
dense_9/StatefulPartitionedCallФ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ< ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ< 
'
_user_specified_namedense_8_input
ѕ	
м
C__inference_dense_11_layer_call_and_return_conditional_losses_10205

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№
Г
&__inference_model_2_layer_call_fn_9090
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_90432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ<::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ<
!
_user_specified_name	input_4"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ћ
serving_default
;
input_40
serving_default_input_4:0џџџџџџџџџ<<
dense_110
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:тј
Ў
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
Я__call__
а_default_save_signature
+б&call_and_return_all_conditional_losses"
_tf_keras_networkє{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_2", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_2", "inbound_nodes": [[["token_and_position_embedding_2", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling1d_2", "inbound_nodes": [[["transformer_block_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["global_average_pooling1d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dropout_10", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dropout_11", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_11", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 60]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ы"ш
_tf_keras_input_layerШ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 60]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
ч
	token_emb
pos_emb
trainable_variables
	variables
regularization_losses
	keras_api
в__call__
+г&call_and_return_all_conditional_losses"К
_tf_keras_layer {"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}

att
ffn

layernorm1

layernorm2
dropout1
dropout2
trainable_variables
	variables
regularization_losses
	keras_api
д__call__
+е&call_and_return_all_conditional_losses"Ѕ
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}

trainable_variables
 	variables
!regularization_losses
"	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layerю{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
щ
#trainable_variables
$	variables
%regularization_losses
&	keras_api
и__call__
+й&call_and_return_all_conditional_losses"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
є

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
к__call__
+л&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
щ
-trainable_variables
.	variables
/regularization_losses
0	keras_api
м__call__
+н&call_and_return_all_conditional_losses"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
і

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
о__call__
+п&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}

7iter

8beta_1

9beta_2
	:decay
;learning_rate'mЃ(mЄ1mЅ2mІ<mЇ=mЈ>mЉ?mЊ@mЋAmЌBm­CmЎDmЏEmАFmБGmВHmГImДJmЕKmЖLmЗMmИ'vЙ(vК1vЛ2vМ<vН=vО>vП?vР@vСAvТBvУCvФDvХEvЦFvЧGvШHvЩIvЪJvЫKvЬLvЭMvЮ"
	optimizer
Ц
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221"
trackable_list_wrapper
Ц
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221"
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
Nlayer_metrics
Olayer_regularization_losses

trainable_variables
Pnon_trainable_variables
	variables
regularization_losses

Qlayers
Rmetrics
Я__call__
а_default_save_signature
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
-
рserving_default"
signature_map
Б
<
embeddings
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
с__call__
+т&call_and_return_all_conditional_losses"
_tf_keras_layerі{"class_name": "Embedding", "name": "embedding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 60000, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
Њ
=
embeddings
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"
_tf_keras_layerя{"class_name": "Embedding", "name": "embedding_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 60, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
[layer_metrics
\layer_regularization_losses
trainable_variables
]non_trainable_variables
	variables
regularization_losses

^layers
_metrics
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
ў
`_query_dense
a
_key_dense
b_value_dense
c_softmax
d_dropout_layer
e_output_dense
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"
_tf_keras_layerъ{"class_name": "MultiHeadAttention", "name": "multi_head_attention_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_2", "trainable": true, "dtype": "float32", "num_heads": 2, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
Ё
jlayer_with_weights-0
jlayer-0
klayer_with_weights-1
klayer-1
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"Т
_tf_keras_sequentialЃ{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
у
paxis
	Jgamma
Kbeta
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"Г
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 32]}}
у
uaxis
	Lgamma
Mbeta
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"Г
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_5", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 32]}}
ч
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
щ
~trainable_variables
	variables
regularization_losses
	keras_api
я__call__
+№&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}

>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15"
trackable_list_wrapper

>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
 layer_regularization_losses
trainable_variables
non_trainable_variables
	variables
regularization_losses
layers
metrics
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
 layer_regularization_losses
trainable_variables
non_trainable_variables
 	variables
!regularization_losses
layers
metrics
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
 layer_regularization_losses
#trainable_variables
non_trainable_variables
$	variables
%regularization_losses
layers
metrics
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_10/kernel
:2dense_10/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
 layer_regularization_losses
)trainable_variables
non_trainable_variables
*	variables
+regularization_losses
layers
metrics
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
 layer_regularization_losses
-trainable_variables
non_trainable_variables
.	variables
/regularization_losses
layers
metrics
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
!:2dense_11/kernel
:2dense_11/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
 layer_regularization_losses
3trainable_variables
non_trainable_variables
4	variables
5regularization_losses
layers
metrics
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
I:G
рд 25token_and_position_embedding_2/embedding_4/embeddings
G:E< 25token_and_position_embedding_2/embedding_5/embeddings
M:K  27transformer_block_2/multi_head_attention_2/query/kernel
G:E 25transformer_block_2/multi_head_attention_2/query/bias
K:I  25transformer_block_2/multi_head_attention_2/key/kernel
E:C 23transformer_block_2/multi_head_attention_2/key/bias
M:K  27transformer_block_2/multi_head_attention_2/value/kernel
G:E 25transformer_block_2/multi_head_attention_2/value/bias
X:V  2Btransformer_block_2/multi_head_attention_2/attention_output/kernel
N:L 2@transformer_block_2/multi_head_attention_2/attention_output/bias
 :  2dense_8/kernel
: 2dense_8/bias
 :  2dense_9/kernel
: 2dense_9/bias
=:; 2/transformer_block_2/layer_normalization_4/gamma
<:: 2.transformer_block_2/layer_normalization_4/beta
=:; 2/transformer_block_2/layer_normalization_5/gamma
<:: 2.transformer_block_2/layer_normalization_5/beta
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
 0
Ё1"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ђlayer_metrics
 Ѓlayer_regularization_losses
Strainable_variables
Єnon_trainable_variables
T	variables
Uregularization_losses
Ѕlayers
Іmetrics
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
'
=0"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Їlayer_metrics
 Јlayer_regularization_losses
Wtrainable_variables
Љnon_trainable_variables
X	variables
Yregularization_losses
Њlayers
Ћmetrics
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Щ
Ќpartial_output_shape
­full_output_shape

>kernel
?bias
Ўtrainable_variables
Џ	variables
Аregularization_losses
Б	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses"ы
_tf_keras_layerб{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 2, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 32]}}
Х
Вpartial_output_shape
Гfull_output_shape

@kernel
Abias
Дtrainable_variables
Е	variables
Жregularization_losses
З	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses"ч
_tf_keras_layerЭ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 2, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 32]}}
Щ
Иpartial_output_shape
Йfull_output_shape

Bkernel
Cbias
Кtrainable_variables
Л	variables
Мregularization_losses
Н	keras_api
ѕ__call__
+і&call_and_return_all_conditional_losses"ы
_tf_keras_layerб{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 2, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 32]}}
ы
Оtrainable_variables
П	variables
Рregularization_losses
С	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
ч
Тtrainable_variables
У	variables
Фregularization_losses
Х	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
о
Цpartial_output_shape
Чfull_output_shape

Dkernel
Ebias
Шtrainable_variables
Щ	variables
Ъregularization_losses
Ы	keras_api
ћ__call__
+ќ&call_and_return_all_conditional_losses"
_tf_keras_layerц{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 2, 32]}}
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ьlayer_metrics
 Эlayer_regularization_losses
ftrainable_variables
Юnon_trainable_variables
g	variables
hregularization_losses
Яlayers
аmetrics
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
њ

Fkernel
Gbias
бtrainable_variables
в	variables
гregularization_losses
д	keras_api
§__call__
+ў&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 32]}}
ќ

Hkernel
Ibias
еtrainable_variables
ж	variables
зregularization_losses
и	keras_api
џ__call__
+&call_and_return_all_conditional_losses"б
_tf_keras_layerЗ{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 32]}}
<
F0
G1
H2
I3"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
йlayer_metrics
 кlayer_regularization_losses
ltrainable_variables
лnon_trainable_variables
m	variables
nregularization_losses
мlayers
нmetrics
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
оlayer_metrics
 пlayer_regularization_losses
qtrainable_variables
рnon_trainable_variables
r	variables
sregularization_losses
сlayers
тmetrics
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
уlayer_metrics
 фlayer_regularization_losses
vtrainable_variables
хnon_trainable_variables
w	variables
xregularization_losses
цlayers
чmetrics
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
шlayer_metrics
 щlayer_regularization_losses
ztrainable_variables
ъnon_trainable_variables
{	variables
|regularization_losses
ыlayers
ьmetrics
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ж
эlayer_metrics
 юlayer_regularization_losses
~trainable_variables
яnon_trainable_variables
	variables
regularization_losses
№layers
ёmetrics
я__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
П

ђtotal

ѓcount
є	variables
ѕ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


іtotal

їcount
ј
_fn_kwargs
љ	variables
њ	keras_api"П
_tf_keras_metricЄ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ћlayer_metrics
 ќlayer_regularization_losses
Ўtrainable_variables
§non_trainable_variables
Џ	variables
Аregularization_losses
ўlayers
џmetrics
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
И
layer_metrics
 layer_regularization_losses
Дtrainable_variables
non_trainable_variables
Е	variables
Жregularization_losses
layers
metrics
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
 layer_regularization_losses
Кtrainable_variables
non_trainable_variables
Л	variables
Мregularization_losses
layers
metrics
ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
 layer_regularization_losses
Оtrainable_variables
non_trainable_variables
П	variables
Рregularization_losses
layers
metrics
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
 layer_regularization_losses
Тtrainable_variables
non_trainable_variables
У	variables
Фregularization_losses
layers
metrics
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
 layer_regularization_losses
Шtrainable_variables
non_trainable_variables
Щ	variables
Ъregularization_losses
layers
metrics
ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
`0
a1
b2
c3
d4
e5"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
layer_metrics
 layer_regularization_losses
бtrainable_variables
non_trainable_variables
в	variables
гregularization_losses
layers
metrics
§__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
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
И
layer_metrics
 layer_regularization_losses
еtrainable_variables
 non_trainable_variables
ж	variables
зregularization_losses
Ёlayers
Ђmetrics
џ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
ђ0
ѓ1"
trackable_list_wrapper
.
є	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
і0
ї1"
trackable_list_wrapper
.
љ	variables"
_generic_user_object
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
&:$ 2Adam/dense_10/kernel/m
 :2Adam/dense_10/bias/m
&:$2Adam/dense_11/kernel/m
 :2Adam/dense_11/bias/m
N:L
рд 2<Adam/token_and_position_embedding_2/embedding_4/embeddings/m
L:J< 2<Adam/token_and_position_embedding_2/embedding_5/embeddings/m
R:P  2>Adam/transformer_block_2/multi_head_attention_2/query/kernel/m
L:J 2<Adam/transformer_block_2/multi_head_attention_2/query/bias/m
P:N  2<Adam/transformer_block_2/multi_head_attention_2/key/kernel/m
J:H 2:Adam/transformer_block_2/multi_head_attention_2/key/bias/m
R:P  2>Adam/transformer_block_2/multi_head_attention_2/value/kernel/m
L:J 2<Adam/transformer_block_2/multi_head_attention_2/value/bias/m
]:[  2IAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/m
S:Q 2GAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/m
%:#  2Adam/dense_8/kernel/m
: 2Adam/dense_8/bias/m
%:#  2Adam/dense_9/kernel/m
: 2Adam/dense_9/bias/m
B:@ 26Adam/transformer_block_2/layer_normalization_4/gamma/m
A:? 25Adam/transformer_block_2/layer_normalization_4/beta/m
B:@ 26Adam/transformer_block_2/layer_normalization_5/gamma/m
A:? 25Adam/transformer_block_2/layer_normalization_5/beta/m
&:$ 2Adam/dense_10/kernel/v
 :2Adam/dense_10/bias/v
&:$2Adam/dense_11/kernel/v
 :2Adam/dense_11/bias/v
N:L
рд 2<Adam/token_and_position_embedding_2/embedding_4/embeddings/v
L:J< 2<Adam/token_and_position_embedding_2/embedding_5/embeddings/v
R:P  2>Adam/transformer_block_2/multi_head_attention_2/query/kernel/v
L:J 2<Adam/transformer_block_2/multi_head_attention_2/query/bias/v
P:N  2<Adam/transformer_block_2/multi_head_attention_2/key/kernel/v
J:H 2:Adam/transformer_block_2/multi_head_attention_2/key/bias/v
R:P  2>Adam/transformer_block_2/multi_head_attention_2/value/kernel/v
L:J 2<Adam/transformer_block_2/multi_head_attention_2/value/bias/v
]:[  2IAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/v
S:Q 2GAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/v
%:#  2Adam/dense_8/kernel/v
: 2Adam/dense_8/bias/v
%:#  2Adam/dense_9/kernel/v
: 2Adam/dense_9/bias/v
B:@ 26Adam/transformer_block_2/layer_normalization_4/gamma/v
A:? 25Adam/transformer_block_2/layer_normalization_4/beta/v
B:@ 26Adam/transformer_block_2/layer_normalization_5/gamma/v
A:? 25Adam/transformer_block_2/layer_normalization_5/beta/v
ц2у
&__inference_model_2_layer_call_fn_9090
&__inference_model_2_layer_call_fn_9716
&__inference_model_2_layer_call_fn_9194
&__inference_model_2_layer_call_fn_9667Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
н2к
__inference__wrapped_model_8189Ж
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *&Ђ#
!
input_4џџџџџџџџџ<
в2Я
A__inference_model_2_layer_call_and_return_conditional_losses_9453
A__inference_model_2_layer_call_and_return_conditional_losses_9618
A__inference_model_2_layer_call_and_return_conditional_losses_8930
A__inference_model_2_layer_call_and_return_conditional_losses_8985Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2п
=__inference_token_and_position_embedding_2_layer_call_fn_9749
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§2њ
X__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_9740
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 2
3__inference_transformer_block_2_layer_call_fn_10098
3__inference_transformer_block_2_layer_call_fn_10061А
ЇВЃ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_9897
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_10024А
ЇВЃ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
­2Њ
:__inference_global_average_pooling1d_2_layer_call_fn_10109
:__inference_global_average_pooling1d_2_layer_call_fn_10120Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
у2р
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_10115
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_10104Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
*__inference_dropout_10_layer_call_fn_10147
*__inference_dropout_10_layer_call_fn_10142Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ш2Х
E__inference_dropout_10_layer_call_and_return_conditional_losses_10132
E__inference_dropout_10_layer_call_and_return_conditional_losses_10137Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
(__inference_dense_10_layer_call_fn_10167Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_10_layer_call_and_return_conditional_losses_10158Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
*__inference_dropout_11_layer_call_fn_10194
*__inference_dropout_11_layer_call_fn_10189Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ш2Х
E__inference_dropout_11_layer_call_and_return_conditional_losses_10184
E__inference_dropout_11_layer_call_and_return_conditional_losses_10179Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
(__inference_dense_11_layer_call_fn_10214Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_11_layer_call_and_return_conditional_losses_10205Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЩBЦ
"__inference_signature_wrapper_9253input_4"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2џќ
ѓВя
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2џќ
ѓВя
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ќ2љ
,__inference_sequential_2_layer_call_fn_10354
+__inference_sequential_2_layer_call_fn_8329
,__inference_sequential_2_layer_call_fn_10341
+__inference_sequential_2_layer_call_fn_8356Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ш2х
G__inference_sequential_2_layer_call_and_return_conditional_losses_10271
F__inference_sequential_2_layer_call_and_return_conditional_losses_8301
F__inference_sequential_2_layer_call_and_return_conditional_losses_8287
G__inference_sequential_2_layer_call_and_return_conditional_losses_10328Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Е2ВЏ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Е2ВЏ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_8_layer_call_fn_10394Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_8_layer_call_and_return_conditional_losses_10385Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_9_layer_call_fn_10433Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_9_layer_call_and_return_conditional_losses_10424Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 Ђ
__inference__wrapped_model_8189=<>?@ABCDEJKFGHILM'(120Ђ-
&Ђ#
!
input_4џџџџџџџџџ<
Њ "3Њ0
.
dense_11"
dense_11џџџџџџџџџЃ
C__inference_dense_10_layer_call_and_return_conditional_losses_10158\'(/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_10_layer_call_fn_10167O'(/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЃ
C__inference_dense_11_layer_call_and_return_conditional_losses_10205\12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_11_layer_call_fn_10214O12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЊ
B__inference_dense_8_layer_call_and_return_conditional_losses_10385dFG3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ< 
Њ ")Ђ&

0џџџџџџџџџ< 
 
'__inference_dense_8_layer_call_fn_10394WFG3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ< 
Њ "џџџџџџџџџ< Њ
B__inference_dense_9_layer_call_and_return_conditional_losses_10424dHI3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ< 
Њ ")Ђ&

0џџџџџџџџџ< 
 
'__inference_dense_9_layer_call_fn_10433WHI3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ< 
Њ "џџџџџџџџџ< Ѕ
E__inference_dropout_10_layer_call_and_return_conditional_losses_10132\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ "%Ђ"

0џџџџџџџџџ 
 Ѕ
E__inference_dropout_10_layer_call_and_return_conditional_losses_10137\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ "%Ђ"

0џџџџџџџџџ 
 }
*__inference_dropout_10_layer_call_fn_10142O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ "џџџџџџџџџ }
*__inference_dropout_10_layer_call_fn_10147O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ "џџџџџџџџџ Ѕ
E__inference_dropout_11_layer_call_and_return_conditional_losses_10179\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 Ѕ
E__inference_dropout_11_layer_call_and_return_conditional_losses_10184\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dropout_11_layer_call_fn_10189O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ}
*__inference_dropout_11_layer_call_fn_10194O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџд
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_10104{IЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Й
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_10115`7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ< 

 
Њ "%Ђ"

0џџџџџџџџџ 
 Ќ
:__inference_global_average_pooling1d_2_layer_call_fn_10109nIЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ "!џџџџџџџџџџџџџџџџџџ
:__inference_global_average_pooling1d_2_layer_call_fn_10120S7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ< 

 
Њ "џџџџџџџџџ О
A__inference_model_2_layer_call_and_return_conditional_losses_8930y=<>?@ABCDEJKFGHILM'(128Ђ5
.Ђ+
!
input_4џџџџџџџџџ<
p

 
Њ "%Ђ"

0џџџџџџџџџ
 О
A__inference_model_2_layer_call_and_return_conditional_losses_8985y=<>?@ABCDEJKFGHILM'(128Ђ5
.Ђ+
!
input_4џџџџџџџџџ<
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Н
A__inference_model_2_layer_call_and_return_conditional_losses_9453x=<>?@ABCDEJKFGHILM'(127Ђ4
-Ђ*
 
inputsџџџџџџџџџ<
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Н
A__inference_model_2_layer_call_and_return_conditional_losses_9618x=<>?@ABCDEJKFGHILM'(127Ђ4
-Ђ*
 
inputsџџџџџџџџџ<
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
&__inference_model_2_layer_call_fn_9090l=<>?@ABCDEJKFGHILM'(128Ђ5
.Ђ+
!
input_4џџџџџџџџџ<
p

 
Њ "џџџџџџџџџ
&__inference_model_2_layer_call_fn_9194l=<>?@ABCDEJKFGHILM'(128Ђ5
.Ђ+
!
input_4џџџџџџџџџ<
p 

 
Њ "џџџџџџџџџ
&__inference_model_2_layer_call_fn_9667k=<>?@ABCDEJKFGHILM'(127Ђ4
-Ђ*
 
inputsџџџџџџџџџ<
p

 
Њ "џџџџџџџџџ
&__inference_model_2_layer_call_fn_9716k=<>?@ABCDEJKFGHILM'(127Ђ4
-Ђ*
 
inputsџџџџџџџџџ<
p 

 
Њ "џџџџџџџџџЙ
G__inference_sequential_2_layer_call_and_return_conditional_losses_10271nFGHI;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ< 
p

 
Њ ")Ђ&

0џџџџџџџџџ< 
 Й
G__inference_sequential_2_layer_call_and_return_conditional_losses_10328nFGHI;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ< 
p 

 
Њ ")Ђ&

0џџџџџџџџџ< 
 П
F__inference_sequential_2_layer_call_and_return_conditional_losses_8287uFGHIBЂ?
8Ђ5
+(
dense_8_inputџџџџџџџџџ< 
p

 
Њ ")Ђ&

0џџџџџџџџџ< 
 П
F__inference_sequential_2_layer_call_and_return_conditional_losses_8301uFGHIBЂ?
8Ђ5
+(
dense_8_inputџџџџџџџџџ< 
p 

 
Њ ")Ђ&

0џџџџџџџџџ< 
 
,__inference_sequential_2_layer_call_fn_10341aFGHI;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ< 
p

 
Њ "џџџџџџџџџ< 
,__inference_sequential_2_layer_call_fn_10354aFGHI;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ< 
p 

 
Њ "џџџџџџџџџ< 
+__inference_sequential_2_layer_call_fn_8329hFGHIBЂ?
8Ђ5
+(
dense_8_inputџџџџџџџџџ< 
p

 
Њ "џџџџџџџџџ< 
+__inference_sequential_2_layer_call_fn_8356hFGHIBЂ?
8Ђ5
+(
dense_8_inputџџџџџџџџџ< 
p 

 
Њ "џџџџџџџџџ< Б
"__inference_signature_wrapper_9253=<>?@ABCDEJKFGHILM'(12;Ђ8
Ђ 
1Њ.
,
input_4!
input_4џџџџџџџџџ<"3Њ0
.
dense_11"
dense_11џџџџџџџџџЗ
X__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_9740[=<*Ђ'
 Ђ

xџџџџџџџџџ<
Њ ")Ђ&

0џџџџџџџџџ< 
 
=__inference_token_and_position_embedding_2_layer_call_fn_9749N=<*Ђ'
 Ђ

xџџџџџџџџџ<
Њ "џџџџџџџџџ< Ш
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_10024v>?@ABCDEJKFGHILM7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ< 
p 
Њ ")Ђ&

0џџџџџџџџџ< 
 Ч
M__inference_transformer_block_2_layer_call_and_return_conditional_losses_9897v>?@ABCDEJKFGHILM7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ< 
p
Њ ")Ђ&

0џџџџџџџџџ< 
  
3__inference_transformer_block_2_layer_call_fn_10061i>?@ABCDEJKFGHILM7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ< 
p
Њ "џџџџџџџџџ<  
3__inference_transformer_block_2_layer_call_fn_10098i>?@ABCDEJKFGHILM7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ< 
p 
Њ "џџџџџџџџџ< 