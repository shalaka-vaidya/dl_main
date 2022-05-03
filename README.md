1. For encoding/self-supervised part
    - we use unlabelled data
    - run the ssltraining.py with right paths to unlabelled data set


2. For supervised/classification part
    -if training from scratch:
        a.  take the checkpoint from part one and plug it into the demo.py file(in get_model())
        b.  execute demo.py file for 30 epochs
    - if training from intermediate checkpoint add the checkpoint on line 54(demo.py)
    - run it by executing the demo.slurm file

3. Final submission has the evaluation script
    - run eval.slurm file with the path of the checkpoint (line 42)
