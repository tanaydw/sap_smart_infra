߫
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
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
delete_old_dirsbool(?
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
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
z
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:9* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:9*
dtype0
z
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

:*
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
?
Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:9*'
shared_nameAdam/dense_27/kernel/m
?
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*
_output_shapes

:9*
dtype0
?
Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_28/kernel/m
?
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:9*'
shared_nameAdam/dense_27/kernel/v
?
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*
_output_shapes

:9*
dtype0
?
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_28/kernel/v
?
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v*
_output_shapes

:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
^

	kernel

regularization_losses
trainable_variables
	variables
	keras_api
^

kernel
regularization_losses
trainable_variables
	variables
	keras_api
d
iter

beta_1

beta_2
	decay
learning_rate	m,m-	v.v/
 

	0
1

	0
1
?
layer_metrics
regularization_losses
non_trainable_variables
trainable_variables
	variables

layers
layer_regularization_losses
metrics
 
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

	0

	0
?
layer_metrics

regularization_losses
non_trainable_variables
trainable_variables
	variables

layers
 layer_regularization_losses
!metrics
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
"layer_metrics
regularization_losses
#non_trainable_variables
trainable_variables
	variables

$layers
%layer_regularization_losses
&metrics
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
 
 

0
1
 

'0
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
4
	(total
	)count
*	variables
+	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

*	variables
~|
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_dense_27_inputPlaceholder*'
_output_shapes
:?????????9*
dtype0*
shape:?????????9
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_27_inputdense_27/kerneldense_28/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_942083
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_27/kernel/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_942214
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_27/kerneldense_28/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_27/kernel/mAdam/dense_28/kernel/mAdam/dense_27/kernel/vAdam/dense_28/kernel/v*
Tin
2*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_942263??
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_942025
dense_27_input
dense_27_942018
dense_28_942021
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalldense_27_inputdense_27_942018*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_9419832"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_942021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_9420022"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall:W S
'
_output_shapes
:?????????9
(
_user_specified_namedense_27_input
?
?
D__inference_dense_28_layer_call_and_return_conditional_losses_942002

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_942015
dense_27_input
dense_27_941992
dense_28_942011
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalldense_27_inputdense_27_941992*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_9419832"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_942011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_9420022"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall:W S
'
_output_shapes
:?????????9
(
_user_specified_namedense_27_input
?
o
)__inference_dense_27_layer_call_fn_942138

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_9419832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????9:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_942038

inputs
dense_27_942031
dense_28_942034
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall?
 dense_27/StatefulPartitionedCallStatefulPartitionedCallinputsdense_27_942031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_9419832"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_942034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_9420022"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_942057

inputs
dense_27_942050
dense_28_942053
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall?
 dense_27/StatefulPartitionedCallStatefulPartitionedCallinputsdense_27_942050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_9419832"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_942053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_9420022"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
!__inference__wrapped_model_941971
dense_27_input9
5sequential_13_dense_27_matmul_readvariableop_resource9
5sequential_13_dense_28_matmul_readvariableop_resource
identity??,sequential_13/dense_27/MatMul/ReadVariableOp?,sequential_13/dense_28/MatMul/ReadVariableOp?
,sequential_13/dense_27/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_27_matmul_readvariableop_resource*
_output_shapes

:9*
dtype02.
,sequential_13/dense_27/MatMul/ReadVariableOp?
sequential_13/dense_27/MatMulMatMuldense_27_input4sequential_13/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_13/dense_27/MatMul?
sequential_13/dense_27/TanhTanh'sequential_13/dense_27/MatMul:product:0*
T0*'
_output_shapes
:?????????2
sequential_13/dense_27/Tanh?
,sequential_13/dense_28/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_28_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_13/dense_28/MatMul/ReadVariableOp?
sequential_13/dense_28/MatMulMatMulsequential_13/dense_27/Tanh:y:04sequential_13/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_13/dense_28/MatMul?
IdentityIdentity'sequential_13/dense_28/MatMul:product:0-^sequential_13/dense_27/MatMul/ReadVariableOp-^sequential_13/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::2\
,sequential_13/dense_27/MatMul/ReadVariableOp,sequential_13/dense_27/MatMul/ReadVariableOp2\
,sequential_13/dense_28/MatMul/ReadVariableOp,sequential_13/dense_28/MatMul/ReadVariableOp:W S
'
_output_shapes
:?????????9
(
_user_specified_namedense_27_input
?
?
.__inference_sequential_13_layer_call_fn_942064
dense_27_input
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_27_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_9420572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????9
(
_user_specified_namedense_27_input
?
?
.__inference_sequential_13_layer_call_fn_942123

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_9420572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
o
)__inference_dense_28_layer_call_fn_942152

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_9420022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
"__inference__traced_restore_942263
file_prefix$
 assignvariableop_dense_27_kernel&
"assignvariableop_1_dense_28_kernel 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate
assignvariableop_7_total
assignvariableop_8_count-
)assignvariableop_9_adam_dense_27_kernel_m.
*assignvariableop_10_adam_dense_28_kernel_m.
*assignvariableop_11_adam_dense_27_kernel_v.
*assignvariableop_12_adam_dense_28_kernel_v
identity_14??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_27_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_28_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_adam_dense_27_kernel_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp*assignvariableop_10_adam_dense_28_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp*assignvariableop_11_adam_dense_27_kernel_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_dense_28_kernel_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13?
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*I
_input_shapes8
6: :::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?&
?
__inference__traced_save_942214
file_prefix.
*savev2_dense_27_kernel_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_27_kernel_read_readvariableop*savev2_dense_28_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*a
_input_shapesP
N: :9:: : : : : : : :9::9:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:9:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :
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
: :$
 

_output_shapes

:9:$ 

_output_shapes

::$ 

_output_shapes

:9:$ 

_output_shapes

::

_output_shapes
: 
?
?
D__inference_dense_28_layer_call_and_return_conditional_losses_942145

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_942083
dense_27_input
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_27_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_9419712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????9
(
_user_specified_namedense_27_input
?
?
D__inference_dense_27_layer_call_and_return_conditional_losses_941983

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:9*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????9:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_942094

inputs+
'dense_27_matmul_readvariableop_resource+
'dense_28_matmul_readvariableop_resource
identity??dense_27/MatMul/ReadVariableOp?dense_28/MatMul/ReadVariableOp?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:9*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMulinputs&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_27/MatMuls
dense_27/TanhTanhdense_27/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_27/Tanh?
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_28/MatMul/ReadVariableOp?
dense_28/MatMulMatMuldense_27/Tanh:y:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_28/MatMul?
IdentityIdentitydense_28/MatMul:product:0^dense_27/MatMul/ReadVariableOp^dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_942105

inputs+
'dense_27_matmul_readvariableop_resource+
'dense_28_matmul_readvariableop_resource
identity??dense_27/MatMul/ReadVariableOp?dense_28/MatMul/ReadVariableOp?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:9*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMulinputs&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_27/MatMuls
dense_27/TanhTanhdense_27/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_27/Tanh?
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_28/MatMul/ReadVariableOp?
dense_28/MatMulMatMuldense_27/Tanh:y:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_28/MatMul?
IdentityIdentitydense_28/MatMul:product:0^dense_27/MatMul/ReadVariableOp^dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
D__inference_dense_27_layer_call_and_return_conditional_losses_942131

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:9*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????9:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
.__inference_sequential_13_layer_call_fn_942114

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_9420382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????9
 
_user_specified_nameinputs
?
?
.__inference_sequential_13_layer_call_fn_942045
dense_27_input
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_27_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_9420382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????9::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????9
(
_user_specified_namedense_27_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
dense_27_input7
 serving_default_dense_27_input:0?????????9<
dense_280
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?c
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
0__call__
*1&call_and_return_all_conditional_losses
2_default_save_signature"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_13", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 57]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_27_input"}}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 57]}, "dtype": "float32", "units": 15, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 57}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 57]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_13", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 57]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_27_input"}}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 57]}, "dtype": "float32", "units": 15, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

	kernel

regularization_losses
trainable_variables
	variables
	keras_api
3__call__
*4&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 57]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 57]}, "dtype": "float32", "units": 15, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 57}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 57]}}
?

kernel
regularization_losses
trainable_variables
	variables
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15]}}
w
iter

beta_1

beta_2
	decay
learning_rate	m,m-	v.v/"
	optimizer
 "
trackable_list_wrapper
.
	0
1"
trackable_list_wrapper
.
	0
1"
trackable_list_wrapper
?
layer_metrics
regularization_losses
non_trainable_variables
trainable_variables
	variables

layers
layer_regularization_losses
metrics
0__call__
2_default_save_signature
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
,
7serving_default"
signature_map
!:92dense_27/kernel
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
'
	0"
trackable_list_wrapper
?
layer_metrics

regularization_losses
non_trainable_variables
trainable_variables
	variables

layers
 layer_regularization_losses
!metrics
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
!:2dense_28/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
"layer_metrics
regularization_losses
#non_trainable_variables
trainable_variables
	variables

$layers
%layer_regularization_losses
&metrics
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
'
'0"
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
?
	(total
	)count
*	variables
+	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
(0
)1"
trackable_list_wrapper
-
*	variables"
_generic_user_object
&:$92Adam/dense_27/kernel/m
&:$2Adam/dense_28/kernel/m
&:$92Adam/dense_27/kernel/v
&:$2Adam/dense_28/kernel/v
?2?
.__inference_sequential_13_layer_call_fn_942123
.__inference_sequential_13_layer_call_fn_942045
.__inference_sequential_13_layer_call_fn_942114
.__inference_sequential_13_layer_call_fn_942064?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_13_layer_call_and_return_conditional_losses_942015
I__inference_sequential_13_layer_call_and_return_conditional_losses_942025
I__inference_sequential_13_layer_call_and_return_conditional_losses_942094
I__inference_sequential_13_layer_call_and_return_conditional_losses_942105?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_941971?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *-?*
(?%
dense_27_input?????????9
?2?
)__inference_dense_27_layer_call_fn_942138?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_27_layer_call_and_return_conditional_losses_942131?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_28_layer_call_fn_942152?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_28_layer_call_and_return_conditional_losses_942145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_942083dense_27_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_941971r	7?4
-?*
(?%
dense_27_input?????????9
? "3?0
.
dense_28"?
dense_28??????????
D__inference_dense_27_layer_call_and_return_conditional_losses_942131[	/?,
%?"
 ?
inputs?????????9
? "%?"
?
0?????????
? {
)__inference_dense_27_layer_call_fn_942138N	/?,
%?"
 ?
inputs?????????9
? "???????????
D__inference_dense_28_layer_call_and_return_conditional_losses_942145[/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
)__inference_dense_28_layer_call_fn_942152N/?,
%?"
 ?
inputs?????????
? "???????????
I__inference_sequential_13_layer_call_and_return_conditional_losses_942015l	??<
5?2
(?%
dense_27_input?????????9
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_13_layer_call_and_return_conditional_losses_942025l	??<
5?2
(?%
dense_27_input?????????9
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_13_layer_call_and_return_conditional_losses_942094d	7?4
-?*
 ?
inputs?????????9
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_13_layer_call_and_return_conditional_losses_942105d	7?4
-?*
 ?
inputs?????????9
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_13_layer_call_fn_942045_	??<
5?2
(?%
dense_27_input?????????9
p

 
? "???????????
.__inference_sequential_13_layer_call_fn_942064_	??<
5?2
(?%
dense_27_input?????????9
p 

 
? "???????????
.__inference_sequential_13_layer_call_fn_942114W	7?4
-?*
 ?
inputs?????????9
p

 
? "???????????
.__inference_sequential_13_layer_call_fn_942123W	7?4
-?*
 ?
inputs?????????9
p 

 
? "???????????
$__inference_signature_wrapper_942083?	I?F
? 
??<
:
dense_27_input(?%
dense_27_input?????????9"3?0
.
dense_28"?
dense_28?????????