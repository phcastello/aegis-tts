"""The archive boundary must reject unsafe or corrupt source material."""
import stat
import zipfile

import pytest

from training.f5_prepare import digest, extract


def make_archive(tmp_path, members):
    archive=tmp_path/'dataset.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for name,content in members:
            z.writestr(name,content)
    return archive


@pytest.mark.parametrize('name',['../escape.wav','/absolute.wav','C:/windows.wav','dir\\escape.wav','x/../../escape.wav'])
def test_reject_unsafe_member(tmp_path,name):
    archive=make_archive(tmp_path,[(name,b'content')])
    with pytest.raises(ValueError,match='unsafe ZIP'):
        extract(archive,tmp_path/'out',expected=digest(archive))
    assert not (tmp_path/'escape.wav').exists()
    assert not (tmp_path/'out').exists()


def test_reject_symbolic_link(tmp_path):
    item=zipfile.ZipInfo('link')
    item.create_system=3
    item.external_attr=(stat.S_IFLNK|0o777)<<16
    archive=make_archive(tmp_path,[(item,b'/etc/passwd')])
    with pytest.raises(ValueError,match='nonregular'):
        extract(archive,tmp_path/'out',expected=digest(archive))


def test_mismatched_archive_hash_never_extracts(tmp_path):
    archive=make_archive(tmp_path,[('dataset/clip.wav',b'content')])
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        extract(archive,tmp_path/'out',expected='0'*64)
    assert not (tmp_path/'out.staging').exists()


def test_reject_duplicate_destination(tmp_path):
    archive=make_archive(tmp_path,[('a/b',b'one'),('a/./b',b'two')])
    with pytest.raises(ValueError,match='duplicate'):
        extract(archive,tmp_path/'out',expected=digest(archive))


def test_crc_is_checked_before_publication(tmp_path):
    archive=make_archive(tmp_path,[('dataset/file.txt',b'CONTENT_UNIQUE')])
    archive.write_bytes(archive.read_bytes().replace(b'CONTENT_UNIQUE',b'CONTEnT_UNIQUE',1))
    with pytest.raises(zipfile.BadZipFile,match='CRC'):
        extract(archive,tmp_path/'out',expected=digest(archive))
    assert not (tmp_path/'out').exists()


def test_bounded_valid_archive_and_private_permissions(tmp_path):
    archive=make_archive(tmp_path,[('dataset/file.txt',b'content')])
    result=extract(archive,tmp_path/'out',expected=digest(archive))
    assert result['crc_all_verified']
    assert (tmp_path/'out/dataset/file.txt').read_bytes()==b'content'
    assert stat.S_IMODE((tmp_path/'out').stat().st_mode)==0o700
    assert stat.S_IMODE((tmp_path/'out/dataset/file.txt').stat().st_mode)==0o600
