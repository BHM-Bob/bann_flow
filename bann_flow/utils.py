

from functools import wraps

import sys, os, time
import chardet, json, glob
import math

import numpy as np

import oneflow as flow

__NO_ERR__ = False
_Params = {
    'flash_attn':False,
}

def TimeCosts(runTimes:int = 1):
    """
    inner is func(times, *args, **kwargs)
    @TimeCosts(9)
    def f(idx, s):
        return s+idx
    print(f(8))\n
    print(f(s = 8))\n
    """
    def ret_wrapper( func ):
        def core_wrapper(*args, **kwargs):
            t0, ret = time.time(), []
            for times in range(runTimes):
                t1 = time.time()
                ret.append(func(times, *args, **kwargs))
                print(f'{times:2d} | {func.__name__:s} used {time.time()-t1:10.3f}s')
            print(f'{func.__name__:s} used {time.time()-t0:10.3f}s in total, {(time.time()-t0)/runTimes:10.3f}s by mean')
            return ret
        return core_wrapper
    return ret_wrapper

def autoparse(init):
    """
    Automatically assign property for __ini__() func
    Example
    ---------
    @autoparse
        def __init__(self, x):
            do something
    fixed from https://codereview.stackexchange.com/questions/269579/decorating-init-for-automatic-attribute-assignment-safe-and-good-practice
    """
    parnames = list(init.__code__.co_varnames[1:])
    defaults = init.__defaults__
    @wraps(init)
    def wrapped_init(self, *args, **kwargs):
        # remove the param who has no default value 
        # but in the end of the parnames
        if 'kwargs' in parnames and parnames[-1] == 'kwargs':
            parnames.remove('kwargs')
        if 'args' in parnames and parnames[-1] == 'args':
            parnames.remove('args')
        # Turn args into kwargs
        kwargs.update(zip(parnames[:len(args)], args))
        # apply default parameter values
        if defaults is not None:
            default_start = len(parnames) - len(defaults)
            for i in range(len(defaults)):
                if parnames[default_start + i] not in kwargs:
                    kwargs[parnames[default_start + i]] = defaults[i]
        # attach attributes to instance
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
        init(self, **kwargs)
    return wrapped_init

def rand_choose_times(choices_range:list[int] = [0,10], times:int = 100):
    randSeq = np.random.randint(low = choices_range[0], high = choices_range[1]+1, size = [times])
    count = [ np.sum(np.equal(randSeq,i)) for i in range(choices_range[0],choices_range[1]+1) ]
    return np.argmax(np.array(count))

def put_err(info:str, ret = None):
    if not __NO_ERR__:
        print(f'\nERR : {sys._getframe().f_code.co_name:s} : {info:s}\n')
    return ret
def put_log(info:str, head = "log", ret = None):
    print(f'\n{head:s} : {sys._getframe().f_code.co_name:s} : {info:s}\n')
    return ret

class MyArgs():
    def __init__(self, args:dict) -> None:
        self.args = dict()
        args = self.get_args(args)
    def get_args(self, args:dict, force_update = True, del_origin = False):
        for arg_name in list(args.keys()):
            if arg_name in self.args and not force_update:
                pass
            else:
                setattr(self, arg_name, args[arg_name])
            if del_origin:
                del args[arg_name]
        return self
    def add_arg(self, arg_name:str, arg_value, force_update = True):
        setattr(self, arg_name, arg_value)
    def toDict(self):
        dic = {}
        for attr in vars(self):
            dic[attr] = getattr(self,attr)
        return dic   

def get_wanted_args(defalut_args:dict, kwargs:dict, del_kwargs = True):
    """
    wanted_args:dict with default value
    localVar = locals()
    """
    return MyArgs(defalut_args).get_args(kwargs, True, del_kwargs)
            
def detect_byte_coding(bits:bytes):
    adchar = chardet.detect(bits[:(1000 if len(bits) > 1000 else len(bits))])['encoding']
    if adchar == 'gbk' or adchar == 'GBK' or adchar == 'GB2312':
        true_text = bits.decode('GB2312', "ignore")
    else:
        true_text = bits.decode('utf-8', "ignore")
    return true_text

def save_json(path:str, obj, encoding:str = 'utf-8', forceUpdate = True):
    if forceUpdate or not os.path.isfile(path):
        json_str = json.dumps(obj, indent=1)
        with open(path, 'w' ,encoding=encoding, errors='ignore') as json_file:
            json_file.write(json_str)
            
def read_json(path:str, encoding:str = 'utf-8', invalidPathReturn = None):
    if os.path.isfile(path):
        with open(path, 'r' ,encoding=encoding, errors='ignore') as json_file:
            json_str = json_file.read()
        return json.loads(json_str)
    return invalidPathReturn

class Mprint:
    """logging tools"""
    def __init__(self, path="log.txt", mode="lazy", cleanFirst=True):
        self.path = path
        self.mode = mode
        self.topString = " "
        self.string = ""

        if cleanFirst:
            with open(path, "w") as f:
                f.write("Mprint : cleanFirst\n")

    def mprint(self, *args):
        string = '[{''} - {''}] '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.topString)
        for item in args:
            if type(item) != "":
                item = str(item)
            string += item + " "

        print(string)

        if self.mode != "lazy":
            with open(self.path, "a+") as f:
                f.write(string + "\n")
        else:
            self.string += string + "\n"

    def logOnly(self, *args):
        string = '[{''} - {''}] '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.topString)
        for item in args:
            if type(item) != "":
                item = str(item)
            string += item + " "

        if self.mode != "lazy":
            with open(self.path, "a+") as f:
                f.write(string + "\n")
        else:
            self.string += string + "\n"

    def exit(self):
        with open(self.path, "a+") as f:
            f.write(self.string)
            
    def __str__(self):
        return f'path={self.path:s}, mode={self.mode:s}, topString={self.topString:s}'
            
class GlobalSettings(MyArgs):
    def __init__(self, mp:Mprint, model_root:str, random_seed:int = 3407):
        # data loading
        self.read = {}# for data reading
        self.batch_size =  64
        self.load_shape = [3, 128, 128]
        # model
        self.model = None
        self.arch = None
        self.lr =  0.01
        self.in_shape =  [64, 3, 128, 128]
        self.out_shape =  [64, 128]
        self.load_db_ratio =  1
        # default var
        self.epochs = 1500
        self.print_freq = 40
        self.test_freq = 5
        self.momentum = 0.9
        self.weight_decay = 1e-4        
        self.seed = random_seed
        self.start_epoch = 0
        self.moco_m = 0.999
        self.moco_k = 3200
        self.byolq_k = 3200
        self.moco_t = 0.07
        self.cos = True        
        # fixed var
        self.paths = {}
        self.data = ''
        self.model_root = model_root
        self.resume_paths = glob.glob(os.path.join(self.model_root,'*.tar'))
        self.resume = self.resume_paths[0] if len(self.resume_paths) > 0 else 'None'
        # other
        self.mp = mp#Mp        
        if self.seed is not None:
            import random
            import oneflow.backends.cudnn as cudnn
            random.seed(self.seed)
            flow.manual_seed(self.seed)
            cudnn.deterministic = True
            
    def toDict(self, printOut = False, mp = None):
        dic = {}
        for attr in vars(self):
            dic[attr] = getattr(self,attr)
        if printOut and mp is not None:
            [ mp.mprint(attr,' : ',dic[attr]) for attr in dic.keys()]
        elif printOut:
            [ print(attr,' : ',dic[attr]) for attr in dic.keys()]
        return dic
    
    def set_resume(self):        
        self.resume_paths = glob.glob(os.path.join(self.model_root,'*.tar'))
        self.resume = self.resume_paths[0] if len(self.resume_paths) > 0 else 'None'

def adjust_learning_rate(optimizer, now_epoch, args):
    """Decay the learning rate based on schedule
    args"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * now_epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if now_epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def format_secs(sumSecs):
    sumHs = int(sumSecs//3600)
    sumMs = int((sumSecs-sumHs*3600)//60)
    sumSs = int(sumSecs-sumHs*3600-sumMs*60)
    return sumHs, sumMs, sumSs

class AverageMeter(object):
    """
    Computes and stores the average and current value
    from FAIR or MAIR 's MoCo
    """
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    """from FAIR or MAIR 's MoCo"""
    def __init__(self, num_batches, meters, prefix="", mp = None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.mp = mp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.mp is None:
            print("\t".join(entries))
        else:
            self.mp.mprint("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
class TimeLast(object):
    def __init__(self):
        self.last_time = time.time()

    def update(self, left_tasks:int, just_done_tasks:int = 1):
        used_time = time.time() - self.last_time
        self.last_time = time.time()
        sum_last_time = left_tasks * used_time / just_done_tasks
        return sum_last_time            
            
def save_checkpoint(epoch, args:GlobalSettings, model, optimizer, loss, other:dict, tailName:str):
    state = {
        "epoch": epoch + 1,
        "arch": args.arch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "args":args.toDict(),
    }
    state.update(other)
    filename = os.path.join(args.modelRoot,
                            f"checkpoint_{tailName:s}_{time.asctime(time.localtime()).replace(':', '-'):s}.pth.tar")
    flow.save(state, filename)

def resume(args, model, optimizer, other:dict = {}):
    if args.resume and os.path.isfile(args.resume):
        args.mp.mprint("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = flow.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            # loc = "cuda:{}".format(args.gpu)
            checkpoint = flow.load(args.resume)  # , map_location=loc)
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        old_losses = checkpoint["loss"]
        args.mp.mprint(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )
        if other:
            other.update()
            for key in other.keys():
                if key in checkpoint:
                    other[key] = checkpoint[key]
        return model, optimizer, old_losses
    else:
        args.mp.mprint("=> no checkpoint found at '{}'".format(args.resume))
        args.mp.logOnly(str(model))
        return model, optimizer, 0
    