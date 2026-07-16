from app.ui import HTML_PAGE, render_ui


def test_ui_is_a_local_dependency_free_manual_test_page() -> None:
    page = render_ui()

    assert page == HTML_PAGE
    assert "Aegis TTS" in page
    assert "https://" not in page
    assert "cdn" not in page.lower()
    assert "fetch('/health')" in page
    assert "fetch('/v1/voices')" in page
    assert "fetch('/v1/synthesize'" in page
    assert "X-API-Key" in page
    assert "sessionStorage" in page
    assert "audio/wav" in page


def test_ui_compares_the_two_references_sequentially() -> None:
    page = render_ui()

    first = page.index("await synthesize(shortPayload")
    second = page.index("await synthesize(longPayload")
    assert first < second
    assert "const shortPayload = currentPayload('giseli-10s')" in page
    assert "const longPayload = {...shortPayload, voice: 'giseli-34s'}" in page
    assert "Promise.all([loadHealth(), loadVoices()])" in page
    assert "Intentional sequential calls" in page


def test_ui_does_not_offer_reference_upload_or_voice_design() -> None:
    page = render_ui().lower()

    assert "type=\"file\"" not in page
    assert "voicedesign" not in page
    assert "customvoice" not in page
