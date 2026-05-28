import logging
import logging.config

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s][%(levelname)s] %(message)s',
            'datefmt': r'%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'cryopros.log',
            'mode': 'a',
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        'CryoPROS': {
            'level': 'INFO',
            'handlers': ['file'],
            'propagate': False
        }
    }
})
logger = logging.getLogger('CryoPROS')
