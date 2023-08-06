from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
import glob
import pandas as pd
# tf.logging.set_verbosity(tf.logging.ERROR)

basedir = "runs"

def load_tf(dirname):
    prefix = basedir + "tboard/VisibleSwimmer-v2/"
    dirname = prefix + dirname
    dirname = glob.glob(dirname + '/*')[0]
    
    ea = event_accumulator.EventAccumulator(dirname, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    dframes = {}
    mnames = ea.Tags()['scalars']
    
    for n in mnames:
        dframes[n] = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "epoch", n.replace('val/', '')])
        dframes[n].drop("wall_time", axis=1, inplace=True)
        dframes[n] = dframes[n].set_index("epoch")
    return pd.concat([v for k,v in dframes.items()], axis=1)

def load_tf_jobs(regex):
    prefix = basedir + "results/"
    job_dirs = glob.glob(prefix + regex)

    rows = []
    for job in job_dirs:
        job_name = os.path.basename(os.path.normpath(job))
        
        # this loads in all the hyperparams from another file,
        # do your own thing here instead
        options = load_json(job + '/opt.json')
        try:
            results = load_tf(job.replace(prefix, ''))
        except:
            continue

        for opt in options:
            results[opt] = options[opt]
        rows.append(results)

    for row in rows:
        row['epoch'] = row.index
        row.reset_index(drop=True, inplace=True)
    df = pd.concat(rows)
    return df