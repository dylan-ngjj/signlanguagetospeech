  *?G??.?@)      p=2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map????{@!??ipG?U@);?p?G?
@1?R??`L@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??
DO
??!???e??=@):??KT???15C?:k=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{???w7??!H?CbgJ%@)??q?????1??A%?$@:Preprocessing2U
Iterator::Model::ParallelMapV2I??-??!??k????)I??-??1??k????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???+H??!?hn,eN??)?f?8???1;a????:Preprocessing2F
Iterator::Model5?+-#???!??qL??@)??n?????1[r???L??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?`obH???!??i??&@)?~j?t???1ԋ#????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?s???z??!ƫ?ʹ???)?s???z??1ƫ?ʹ???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangeu?BY??z?!*??h?i??)u?BY??z?1*??h?i??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?b.?z?!??j(???)?b.?z?1??j(???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::TensorSlicenYk(?w?!;Ǚ?????)nYk(?w?1;Ǚ?????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::TensorSlices?m?B<b?!? ?w@4??)s?m?B<b?1? ?w@4??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.