import os

class Config(object):

    def __init__(self):
       
        self.dataset             = ""
        self.log_dir             = "log_dir"
        self.seed                = 3000
        self.save_dir            = "Results"
        self.mode                = "hard"
               
        self.device              = "cuda:0"
        self.validate            = True
        
        self.upscaling           = 2
        
        self.batch_size          = 64
        self.epochs              = 3700
        self.learning_rate       = 1e-4
        self.workers             = 0
            
        self.width               = 32
        self.height              = 32   
        
        self.validate_every      = 10          
        self.log_every_iter      = 1000
        self.validate_every_iter = 100
        self.save_every_iter     = 1000
        
        self.acc_steps           = 2
        self.num_bande           = 30
        
        self.save_every          = 20
        self.resume_training     = False
        self.checkpoint_path     = ""
