def get_unique_run_id():
    import os
    folders = os.listdir(".")
    run_id = get_random_run_id()
    while run_id in folders:
        run_id = get_random_run_id()
    
    return run_id

def get_random_run_id():
    import uuid
    id = uuid.uuid1()
    first_eight = str(id)[0:8]
    run_id = f"run_{first_eight}"
    return run_id
