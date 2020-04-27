speedup
=======

This directory contains notebooks used to check to analyze how the reading
and processing data performs, allowing us to swap out back-end storage,
underlying disk, or underlying libraries behind the same operations.

### Test Box

To work with this repo, go to the test server behind the jumpbox:
```
ssh -i ~/.ssh/celsius-dev.pem -t -l ubuntu -J ubuntu@34.234.63.81 -L 8888:localhost:8888 10.174.1.111
```

Note 1: Switch port 8888 to another port if you have collisions with others on the machine.

Note 2: If you make an alternate user on that box (`adduser YOURNAME`) switch the `-l` option above to your username,
but NOT the `ubuntu@` on the jumpbox.

### Test Data

The notebooks expect data is mounted at:
- `/mnt/ebs_gp2`
- `/mnt/ssd`

The data above comes from s3://celsius-external-speedup.

If you are not on the test box above, create those mount points, or manually modify the notebooks
to point to a different location.

### Usage:

To use this repo, clone it, and put the following other repos next-to ctxbio on your test instance:
- celsius-utils
- scannotate
- cesium3 (only the client/ package is used)

Then run this once to initialize a virtualenv:
```
./setup.sh
```

In any given shell session, run this to re-initialize the env and PYTHONPATH
```
env.sh
```

To run the notebooks, use jupyter in the initialized environment:
```
jupyter notebook speedup.ipynb
```
or
```
jupyter notebook speedup-profiled.ipynb
```

