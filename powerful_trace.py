'''
Author: physico
Date: 2022-10-27 11:33:51
LastEditTime: 2022-11-09 20:47:18
FilePath: /mmdetection3/powerful_trace.py
Description: 
Function List: 
'''
# 代码验证

import trace
import os
import pickle
import pandas as pd
from typing import List
import importlib
import mmdet
import mmengine

class MyTrace(object):
    """trace what you want"""
    def __init__(self,ignoremods:List[str],ignoredirs:List[str],filtermods:List[str],renamemods:List[str],filename:str,funcname:str,if_annotation:bool):
        self.outfile = filename
        self.func = funcname
        self.filtermods = filtermods
        self.renamemods = renamemods
        self.annotation = if_annotation
        self._init_outfile()
        self.trace = trace.Trace(
            ignoremods,
            ignoredirs,
            countfuncs=True,
            outfile=self.outfile)

    def _init_outfile(self):
        if not os.path.exists("./trace_file"):
            os.mkdir("trace_file")
        self.outfile  = "./trace_file" + "/" + self.outfile
        if not os.path.exists(self.outfile):
            with open(self.outfile,"wb"):
                pass     
        else:
            os.remove(self.outfile)
            with open(self.outfile,"wb"):
                    pass     

    def _get_trace_pkl(self):
        self.trace.run(self.func)
        r = self.trace.results()
        r.write_results(show_missing=False, coverdir=None)   

    def _result_loader(self):
        pkl_file = open(self.outfile, 'rb')
        pkl_data = pickle.load(pkl_file)
        pkl_file.close()
        return pkl_data

    def _result_clean(self,data_list):
        """maybe repeatedly delete 'j' will throw an exception"""
        for i in data_list.copy():
            for j in self.filtermods:
                try:
                    if j in i[0]:
                        data_list.remove(i)
                except:
                    pass
        return data_list

    def _result_rename(self,data_list):
        for n,i in enumerate(data_list):
            for j in self.renamemods:
                if j in i[0]:
                    data_list[n][0] = i[0].split(j+"/")[-1]
        return data_list
    
    @classmethod
    def debug_print(filename="trace_file/trace_result.pkl"):
        if os.path.exists(filename):
            pkl_file = open(filename, 'rb')
            pkl_data = pickle.load(pkl_file)
            pkl_file.close()
        else:
            raise(IOError,f"you should confirm {filename} is existed")


    def get_pkl_list(self)->list:
        self._get_trace_pkl()
        pkl_result = self._result_loader()
        pkl_list = [list(i[0]) for i in pkl_result[1].items()]
        pkl_list = self._result_clean(pkl_list)
        pkl_list = self._result_rename(pkl_list)
        print("get_pkl_list success!")
        return pkl_list

    def get_result_csv(self,data_list:list):
        df = pd.DataFrame(columns=['path','file_name',"func_name","annotation"])
        for i in data_list:
            df.loc[df.shape[0]] = dict(zip(df.columns,i))
        
        if self.annotation:
            self.get_annotation(df)
        df.to_csv(self.outfile.split(".pkl")[0]+".csv")
    
    @staticmethod
    def run():
        pkl_list = my_trace.get_pkl_list()
        pkl_list = custom_filter(pkl_list)
        print(pkl_list)
        # output the result
        my_trace.get_result_csv(pkl_list)

    def get_annotation(self,df):
        df["annotation"] = "None"
        for i in range(len(df)):
            file_name = df.iloc[i,0]
            func_name = df.iloc[i,2]
            import_from_file = 0
            if type(file_name)==str:
                file_import_name = file_name.split(".py")[0]
                file_import_name = file_import_name.replace("/",".")

            if "mmengine" in file_import_name and type(file_import_name)==str:
                import_from_file = "mmengine"+ file_import_name.split("mmengine")[-1] 

                import_module = importlib.import_module(import_from_file)
                assert type(import_module)==type(importlib)
                if func_name.split(".")[0] in dir(import_module) and type(import_from_file)==str:
                    function = eval(import_from_file+"."+func_name)
                    assert callable(function)
                    
                    if hasattr(function,'__doc__') and (function.__doc__ != ""):
                        df["annotation"][i] = function.__doc__        

            _file_import_name = file_import_name
            if "mmdet" in _file_import_name.split(".") and type(file_import_name)==str:
                import_from_file = "mmdet"+ file_import_name.split("mmdet")[-1] 
                import_module = importlib.import_module(import_from_file)
                assert type(import_module)==type(importlib)
                if func_name.split(".")[0] in dir(import_module) and type(import_from_file)==str:
                    try:
                        function = eval(import_from_file+"."+func_name)
                    except:
                        function = eval(import_from_file)

                    assert callable(function)
                    if hasattr(function,'__doc__') and (function.__doc__ != ""):
                        df["annotation"][i] = function.__doc__
            
        # df.to_csv("trace_file/trace_result2.csv")
if __name__ == "__main__":

    """ Add the function what you want to be traced here
        Don't put 'import' in main()!!!!!
    """
    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import register_all_modules
    from mmdet.registry import VISUALIZERS
    import mmcv

    def main():
        register_all_modules()
        config_file = './checkpoints/yolov3_mobilenetv2_8xb24-320-300e_coco.py'
        checkpoint_file = './checkpoints/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
        model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
        image = mmcv.imread( "demo/demo.jpg", channel_order='rgb')
        result = inference_detector(model, image)
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta
        visualizer.add_datasample(
            'result',
            image,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
        )

    def custom_filter(data_list):
        """Some filter operations cannot be combined with the previous operations, we must do it in the end.
           Whether to add this step depend on the final result.
        """
        for i in data_list.copy():
            if "python3.7" in i[0] and "mmengine" not in i[0]:
                data_list.remove(i)
        for n,i in enumerate(data_list):
            if "site-packages" in i[0]:
                data_list[n][0] = i[0].split("site-packages/")[-1]
        print("custom_filter success!")
        return data_list

    # create a Trace object, telling it what to ignore
    funcname = "main()"
    ignoremods=["tqdm","matplotlib","PIL","tkinter"]
    ignoredirs=["/home/sanbu/anaconda3/envs/mmlab2/lib/python3.7/site-packages/PIL",
                "/home/sanbu/anaconda3/envs/mmlab2/lib/python3.7/site-packages/matplotlib"]
    filtermods=["_bootstrap","array_function","matplotlib","PIL"]
    renamemods=["mmdetection3","envs"]

    # build mytrace instance
    my_trace = MyTrace(ignoremods,ignoredirs,filtermods,renamemods,filename = "trace_result.pkl",
                        funcname = funcname, if_annotation=True)
    my_trace.run()
