PLAYER_ID = None # your player_id
API_KEY = None # your api_key
TIG_WORKER_PATH = None # path to executable tig-worker
TIG_ALGORITHMS_FOLDER = None # path to tig-algorithms folder
API_URL = "https://mainnet-api.tig.foundation"

if PLAYER_ID is None or API_KEY is None or TIG_WORKER_PATH is None or TIG_ALGORITHMS_FOLDER is None:
    raise Exception("Please set the PLAYER_ID, API_KEY, and TIG_WORKER_PATH, TIG_ALGORITHMS_FOLDER variables in 'tig-benchmarker/master/config.py'")

PORT = 5115
JOBS = dict(
    satisfiability=dict(
        schnoing=dict(
            benchmark_duration=10000, # amount of time to run the benchmark in milliseconds
            wait_slave_duration=5000, # amount of time to wait for slaves to post solutions before submitting
            num_jobs=1, # number of jobs to create. each job will sample its own difficulty
            weight=1.0, # weight of jobs for this algorithm. more weight = more likely to be picked
        )
    ),
    vehicle_routing=dict(
        clarke_wright=dict(
            benchmark_duration=10000,
            wait_slave_duration=5000,
            num_jobs=1,
            weight=1.0,
        )
    ),
    knapsack=dict(
        dynamic=dict(
            benchmark_duration=10000,
            wait_slave_duration=5000,
            num_jobs=1,
            weight=1.0,
        )
    ),
    vector_search=dict(
        optimal_ann=dict(
            benchmark_duration=30000, # recommend a high duration
            wait_slave_duration=30000, # recommend a high duration
            num_jobs=1,
            weight=1.0,
        )
    ),
)