import logging
import sys
import inspect

# logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')
# logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
#     format='%(asctime)s %(filename)s:%(lineno)d - %(message)s')
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='- %(message)s')

def prefix():
    callerframerecord = inspect.stack()[2]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    txt = '{}:{}'.format(info.filename,info.lineno)
    return txt

def info(msg):
    logging.info("{} {}".format(prefix(), msg))

# display once each message
onceMarkers = []
def once(msg):
    marker = prefix()
    if (marker not in onceMarkers):
        onceMarkers.append(marker)
        logging.info("{} {}".format(marker, msg))

# input x must be tensor
def describe(x):
    logging.info("{} Type {},Shape{},Value:\n{}".format(prefix(),x.type(),x.shape,x))
# input x must be tensor
describeOnceMarkers = []
def describeOnce(x):
    marker = prefix()
    if (marker not in describeOnceMarkers):
        describeOnceMarkers.append(marker)
        logging.info("{} Type {},Shape{},Value:\n{}".format(prefix(),x.type(),x.shape,x))


    