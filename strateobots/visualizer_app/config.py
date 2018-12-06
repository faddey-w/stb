GAME_MAX_TICKS = 2000

RANDOM_MATCH_SETTINGS = (
    'RvR', 'TvT', 'LvL',
    'RvT', 'RvL', 'LvT',
    'RTLvRTL',
    'RRRvRRR',
)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'stderr': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        }
    },
    'filters': {},
    'formatters': {
        'default': {
            'format': '%(levelname)s:%(name)s:%(message)s',
        }
    },
    'loggers': {
            '': {
                'handlers': ['stderr'],
                'level': 'WARNING',
            },
            'strateobots': {
                'handlers': ['stderr'],
                'level': 'INFO',
                'propagate': False,
            }
        }
}

DEBUG_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'stderr': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        }
    },
    'filters': {},
    'formatters': {
        'default': {
            'format': '%(levelname)s:%(name)s:%(message)s',
        }
    },
    'loggers': {
            '': {
                'handlers': ['stderr'],
                'level': 'INFO',
            },
            'strateobots': {
                'handlers': ['stderr'],
                'level': 'DEBUG',
                'propagate': False,
            }
        }
}

