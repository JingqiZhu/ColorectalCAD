import os
import requests

# Video number, base URL and save path
video_num = {'adenoma': 40,
             'serrated': 15,
             'hyperplastic': 21,}
base_url = {
    'adenoma_wl': 'https://www.depeca.uah.es/colonoscopy_dataset/Adenoma/adenoma_{:02d}/videos/WL.mp4',
    'serrated_wl': 'https://www.depeca.uah.es/colonoscopy_dataset/Serrated/serrated_{:02d}/videos/WL.mp4',
    'hyperplastic_wl': 'https://www.depeca.uah.es/colonoscopy_dataset/Hyperplastic/hyperplastic_{:02d}/videos/WL.mp4',
    'adenoma_nbi': 'https://www.depeca.uah.es/colonoscopy_dataset/Adenoma/adenoma_{:02d}/videos/NBI.mp4',
    'serrated_nbi': 'https://www.depeca.uah.es/colonoscopy_dataset/Serrated/serrated_{:02d}/videos/NBI.mp4',
    'hyperplastic_nbi': 'https://www.depeca.uah.es/colonoscopy_dataset/Hyperplastic/hyperplastic_{:02d}/videos/NBI.mp4',
    }
base_save_path = {
    'adenoma_wl': '../data/colonoscopic/Adenoma/adenoma_{:02d}_WL.mp4',
    'serrated_wl': '../data/colonoscopic/Serrated/serrated_{:02d}_WL.mp4',
    'hyperplastic_wl': '../data/colonoscopic/Hyperplastic/hyperplastic_{:02d}_WL.mp4',
    'adenoma_nbi': '../data/colonoscopic/Adenoma/adenoma_{:02d}_NBI.mp4',
    'serrated_nbi': '../data/colonoscopic/Serrated/serrated_{:02d}_NBI.mp4',
    'hyperplastic_nbi': '../data/colonoscopic/Hyperplastic/hyperplastic_{:02d}_NBI.mp4',
    }

# Download function
def download_colonoscopic_data(base_url, base_save_path, video_num):
    for i in range(1, video_num + 1):
        url = base_url.format(i)
        save_path = base_save_path.format(i)

        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Send a GET request to the URL with SSL verification disabled
        response = requests.get(url, stream=True, verify=False)

        # Check if the request was successful
        if response.status_code == 200:
            # Open a local file with write-binary mode
            with open(save_path, 'wb') as file:
                # Stream the file from the response
                for chunk in response.iter_content(chunk_size=1024):
                    # Write the chunk to the file
                    file.write(chunk)
            print(f"File downloaded successfully and saved as {save_path}")
        else:
            print(f"Failed to download the file from {url}. Status code: {response.status_code}")

# Download videos
download_colonoscopic_data(base_url['adenoma_wl'], base_save_path['adenoma_wl'], video_num['adenoma'])
download_colonoscopic_data(base_url['adenoma_nbi'], base_save_path['adenoma_nbi'], video_num['adenoma'])
download_colonoscopic_data(base_url['serrated_wl'], base_save_path['serrated_wl'], video_num['serrated'])
download_colonoscopic_data(base_url['serrated_nbi'], base_save_path['serrated_nbi'], video_num['serrated'])
download_colonoscopic_data(base_url['hyperplastic_wl'], base_save_path['hyperplastic_wl'], video_num['hyperplastic'])
download_colonoscopic_data(base_url['hyperplastic_nbi'], base_save_path['hyperplastic_nbi'], video_num['hyperplastic'])