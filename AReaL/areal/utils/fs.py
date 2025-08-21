import getpass
import os


def get_user_tmp():
    user = getpass.getuser()
    # Use NFS path instead of hardcoded /storage path
    user_tmp = os.path.join("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yanghaocheng04/ASearcher", ".cache", "realhf", user)
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp