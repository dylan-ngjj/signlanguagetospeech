?  *	?????M?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMape?X???!?=???I@)????1${w?ZH@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?X?? ??!y??~?E@)c?=yX??1????D@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??ݓ????!??}q$@)(~??k	??1o
}?h@:Preprocessing2U
Iterator::Model::ParallelMapV2??6???!}??}`??)??6???1}??}`??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenatew-!?l??!$s6R???)Q?|a2??10#??3??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenateŏ1w-!??!???y?	??)?HP???1??Y??Y??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?5?;Nё?!?Q????)??@??ǈ?1Ib?~Q ??:Preprocessing2F
Iterator::ModelHP?sע?!???B??@)?j+??݃?1~???????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch/n????!????i???)/n????1????i???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??"??~??!h?G??J@)y?&1?|?1s?I??b??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?g??s?u?!j???s???)?g??s?u?1j???s???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?????g?!??	?\??)?????g?1??	?\??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice-C??6Z?!???F???)-C??6Z?1???F???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensorǺ???V?!\t?}???)Ǻ???V?1\t?}???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensora2U0*?S?!*???????)a2U0*?S?1*???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q=/"A?B??"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.