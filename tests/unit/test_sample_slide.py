from pathbench.utils.test_samples import download_sample_slide

def test_download_sample_slide():
    slide_path = download_sample_slide()
    # Check whether slide was downloaded and has correct file extension
    assert os.path.exists(slide_path), "Sample slide was not downloaded."
    assert slide_path.endswith(".svs"), "Downloaded slide does not have .svs extension."
    import lazyslide as ls
    wsi = ls.open_wsi(slide_path)  # Check if slide can be opened without error
    