PS C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS> & C:/Users/batoul-kanaan/CNN/CNN_RF_NIDS/.venv/Scripts/Activate.ps1
python scripts/run_kdd99.py --data_dir data --binary true --feature_k 18 --max_rows 500000 --cv_folds 5
python scripts/run_kdd99.py --data_dir data --binary true --feature_k 18 --max_rows 500000

(.venv) PS C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS> python scripts/run_kdd99.py --data_dir data --binary true --feature_k 18 --max_rows 50000
>> python scripts/run_unsw_nb15.py --data_dir data/unsw-nb15 --binary true --feature_k 22 --max_rows 50000
2026-02-24 14:35:46.212653: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-24 14:35:47.195842: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Traceback (most recent call last):
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\scripts\run_kdd99.py", line 60, in <module>
    main()
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\scripts\run_kdd99.py", line 45, in main
    run_single_split(
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\pipeline.py", line 144, in run_single_split
    df, y, meta = load_kdd99(data_dir=data_dir, binary=binary, max_rows=max_rows, seed=seed)
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\data.py", line 292, in load_kdd99
    df[col] = df[col].apply(_decode_if_bytes)
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\.venv\lib\site-packages\pandas\core\series.py", line 4936, in apply
    return SeriesApply(
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\.venv\lib\site-packages\pandas\core\apply.py", line 1422, in apply
    return self.apply_standard()
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\.venv\lib\site-packages\pandas\core\apply.py", line 1502, in apply_standard
    mapped = obj._map_values(
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\.venv\lib\site-packages\pandas\core\base.py", line 925, in _map_values
    return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\.venv\lib\site-packages\pandas\core\algorithms.py", line 1743, in map_array
    return lib.map_infer(values, mapper, convert=convert)
  File "pandas/_libs/lib.pyx", line 2999, in pandas._libs.lib.map_infer
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\data.py", line 106, in _decode_if_bytes
    if isinstance(x, (bytes, bytearray)):
KeyboardInterrupt
(.venv) PS C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS> python scripts/run_kdd99.py --data_dir data --binary true --feature_k 18 --max_rows 50000
2026-02-24 14:36:34.048562: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-24 14:36:35.535918: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-24 14:37:23.567341: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.  
2026-02-24 14:37:23.845090: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_15}}
2026-02-24 14:37:34.482594: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\plots.py:55: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
  ax.set_xticklabels(names, rotation=30, ha='right')
(.venv) PS C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS> python scripts/run_unsw_nb15.py --data_dir data/unsw-nb15 --binary true --feature_k 22 --max_rows 50000
2026-02-24 14:38:52.292435: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-24 14:38:53.964358: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-24 14:38:55.273035: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.  
2026-02-24 14:38:55.349888: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_15}}
2026-02-24 14:40:37.941280: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\plots.py:55: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
  ax.set_xticklabels(names, rotation=30, ha='right')
(.venv) PS C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS> python scripts/run_all.py --data_dir data --binary true
2026-02-24 14:50:38.892439: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-24 14:50:39.904875: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Traceback (most recent call last):
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\scripts\run_all.py", line 62, in <module>
    main()
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\scripts\run_all.py", line 34, in main
    run_single_split(
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\pipeline.py", line 144, in run_single_split
    df, y, meta = load_kdd99(data_dir=data_dir, binary=binary, max_rows=max_rows, seed=seed)
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\data.py", line 292, in load_kdd99
    df[col] = df[col].apply(_decode_if_bytes)
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\.venv\lib\site-packages\pandas\core\series.py", line 4936, in apply
    return SeriesApply(
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\.venv\lib\site-packages\pandas\core\apply.py", line 1422, in apply
    return self.apply_standard()
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\.venv\lib\site-packages\pandas\core\apply.py", line 1502, in apply_standard
    mapped = obj._map_values(
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\.venv\lib\site-packages\pandas\core\base.py", line 925, in _map_values
    return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\.venv\lib\site-packages\pandas\core\algorithms.py", line 1743, in map_array
    return lib.map_infer(values, mapper, convert=convert)
  File "pandas/_libs/lib.pyx", line 2999, in pandas._libs.lib.map_infer
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\data.py", line 106, in _decode_if_bytes
    if isinstance(x, (bytes, bytearray)):
KeyboardInterrupt
(.venv) PS C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS> python scripts/run_all.py --data_dir data --binary true --max_rows 50000
2026-02-24 14:51:26.537179: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-24 14:51:27.492057: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-24 14:52:06.745923: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.  
2026-02-24 14:52:06.821228: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_15}}
2026-02-24 14:52:17.200458: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\plots.py:55: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
  ax.set_xticklabels(names, rotation=30, ha='right')
Traceback (most recent call last):
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\scripts\run_all.py", line 62, in <module>
    main()
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\scripts\run_all.py", line 47, in main
    run_single_split(
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\pipeline.py", line 146, in run_single_split
    df, y, meta = load_unsw_nb15(
  File "C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS\src\data.py", line 331, in load_unsw_nb15
    raise FileNotFoundError('UNSW-NB15 CSV files not found in data_dir.')
FileNotFoundError: UNSW-NB15 CSV files not found in data_dir.
(.venv) PS C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS> ^C
(.venv) PS C:\Users\batoul-kanaan\CNN\CNN_RF_NIDS> python scripts/run_all.py --data_dir data --binary true --max_rows 50000
ing-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-24 14:57:46.689757: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
ing-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-24 14:57:46.689757: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.  
2026-02-24 14:57:46.768465: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_15}}
2026-02-24 14:57:57.540890: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
2026-02-24 14:58:50.188262: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_15}}
2026-02-24 15:00:33.793382: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}