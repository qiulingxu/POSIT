from tracemalloc import start
import GPUtil
import os
import sys
import time
from multiprocessing import Pool

class manager():
    def __init__(self, nums ,gpus= 8, start_protect=120, exempt=None, mem_est = 0.2, load_est = 0.2):
        self.exempt = exempt 
        self.gpus = gpus
        self.start_protect = start_protect
        self.gpu_lock = [time.time() for i in range(gpus)]
        self.mem_est = mem_est
        self.load_est = load_est
        self.nums = nums
        self.pool = Pool(nums)
        self.curr_proc = []

    def get_exempt_gpus(self):
        exempt = []
        if self.exempt is not None:
            for gpu in self.exempt:
                exempt.append(gpu)
        t = time.time()
        for i in range(self.gpus):
            if self.gpu_lock[i]>t:
                if i not in exempt:
                    exempt.append(i)
        return exempt

    def start_file(self, pyfile, args):
        for arg in args:
            while True:
                self.check_proc()
                if len(self.curr_proc)<self.nums:
                    try:        
                        exempt = self.get_exempt_gpus()
                        deviceID = GPUtil.getFirstAvailable(excludeID=exempt, order = 'first', maxLoad=1-self.load_est, maxMemory=1-self.mem_est, attempts=1, interval=60, verbose=False)[0]
                        print("use GPU", deviceID)
                        self.run_py(run_py_file, deviceID, pyfile, arg)
                        break
                    except:
                        print("no GPU")
                else:
                    print("task full", len(self.curr_proc))
                time.sleep(5)              
        self.pool.close()
        self.pool.join()
        
    def start_func(self, func, kargs):
        for karg in kargs:
            while True:
                self.check_proc()
                if len(self.curr_proc)<self.nums:
                    try:        
                        exempt = self.get_exempt_gpus()
                        deviceID = GPUtil.getFirstAvailable(excludeID=exempt, order = 'first', maxLoad=1-self.load_est, maxMemory=1-self.mem_est, attempts=1, interval=60, verbose=False)[0]
                        print("use GPU", deviceID)
                        self.run_func(run_py_func, deviceID, func, karg)
                        break
                    except:
                        print("no GPU")
                else:
                    print("task full", len(self.curr_proc))
                time.sleep(5)     
        self.pool.close()
        self.pool.join()

    def check_proc(self):
        curr_proc = []
        for token in self.curr_proc:
            if not token.ready():
                curr_proc.append(token)
        self.curr_proc = curr_proc

    def run_py(self, func, deviceID, python, arg):
        self.gpu_lock[deviceID] = time.time() + self.start_protect
        print("choose GPU {} , python {}, arg {}".format(deviceID, python, arg))
        args = (deviceID,python, arg)
        kargs = {}
        token = self.pool.apply_async(func, args, kargs)
        self.curr_proc.append(token)
        return token

    def run_func(self, func, deviceID, python, karg):
        self.gpu_lock[deviceID] = time.time() + self.start_protect
        print("choose GPU {} , func {}, karg {}".format(deviceID, func, karg))
        args = (deviceID, python, karg)
        kargs = {}
        token = self.pool.apply_async(func, args, kargs)
        self.curr_proc.append(token)
        return token


def run_py_file(deviceID, python, arg):
    if arg is None:
        args = " ".join(sys.argv[1:])
    else:
        args = arg    
    cmd = 'CUDA_VISIBLE_DEVICES={} python ./{} {}'\
        .format(deviceID, python, args) #--skip-exist
    ret = os.system(cmd)
    if ret !=0:
        open("failed.txt","a").write(str(cmd) + " " + str(ret) + "\r\n")
    return ret

def run_py_func(deviceID, func, karg):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(deviceID)
    try:
        func(**karg)
        ret = 0
    except Exception as e:
        print(e)
        ret = str(e)
    if ret !=0:
        open("failed.txt","a").write(str(karg) + " " + str(ret) + "\r\n")
    return ret

def temp():
    print("I am in")

if __name__=="__main__":
    open("temp.py","w").write("""print("I am In")""")
    manage = manager(16, gpus=8, start_protect=10, exempt=[0,1],mem_est=0.5, load_est=0.01)
    
    #manage.start_file("temp.py", args =["" for i in range(100)])
    manage.start_func(temp, kargs =[{} for i in range(100)])