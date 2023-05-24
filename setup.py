from setuptools import setup, find_packages
setup(
    name = 'gcbh2',
    version = '0.0.1',
    packages = find_packages(),
    scripts=['./bondnet/scripts/training/controller_train_hydro.py',
            './bondnet/scripts/training/controller_train.py',
            './bondnet/scripts/training/lightning_bayes_opt.py',
            './bondnet/scripts/training/lightning_train.py',
            './bondnet/scripts/training/lightning_controller.py',
            './bondnet/scripts/training/lightning_generate_settings.py',
            './bondnet/scripts/training/generate_settings.py',
            './bondnet/scripts/training/launch_sbatch.py',
             ],
)
