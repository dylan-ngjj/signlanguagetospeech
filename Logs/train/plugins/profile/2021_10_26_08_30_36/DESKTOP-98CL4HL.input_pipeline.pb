  *	?????1?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj?t???!?YQ2U?U@)??v????1 v5?6U@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map????S??!1=)$A@)I??&??1?,f??i@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?c?ZB??!?????@)K?46??1?\?j@:Preprocessing2U
Iterator::Model::ParallelMapV2"??u????!?] ?ґ??)"??u????1?] ?ґ??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenatea??+e??!???2???)??ͪ?Ֆ?1???F???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???<,Ԛ?!e5??o???)??_vO??1Hv0?O??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate'???????!??償&??)ˡE?????1h????m??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchvq?-??!?ґ?YS??)vq?-??1?ґ?YS??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??(\????!P?M_?V@)K?=?U??1?|D.???:Preprocessing2F
Iterator::Model???JY???!???
?1@)???S㥋?1?<????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?r?!??u?:~??)HP?s?r?1??u?:~??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????Mbp?!2???j???)????Mbp?12???j???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensor?~j?t?X?!?X5<??)?~j?t?X?1?X5<??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice????MbP?!2???j???)????MbP?12???j???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor??H?}M?!`H@???)??H?}M?1`H@???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.