"""Durable persistence: a session (persona + tracker + memory) survives a 'restart'."""

import asyncio
import tempfile
import time

from api.persistence import PersistenceStore
from api.sessions import SessionStore, _SESSION_TTL_SECONDS
from careerpilot.bootstrap import build_shared_resources


def _resources():
    return asyncio.get_event_loop().run_until_complete(build_shared_resources())


async def test_session_rehydrates_after_restart():
    res = await build_shared_resources()
    path = tempfile.mktemp(suffix=".db")
    s1 = SessionStore(res, PersistenceStore(path))
    sess = s1.create("riya")
    tok = sess.token
    sess.sql_client.replace_applications([{"company": "Acme", "role": "GenAI Eng", "status": "Offer"}])
    s1.persist_applications(sess)
    sess.orchestrator.memory.add_user("hi")
    sess.orchestrator.memory.add_assistant("hello")
    s1.persist_memory(sess)

    # Fresh store object (empty in-memory) pointed at the same file = a restart.
    s2 = SessionStore(res, PersistenceStore(path))
    reh = s2.get(tok)
    assert reh is not None and reh.persona_id == "riya"
    rows = await reh.sql_client.run_query("SELECT status FROM applications")
    assert rows == [{"status": "Offer"}]
    assert reh.orchestrator.memory.as_messages()[0]["content"] == "hi"


async def test_unknown_token_returns_none():
    res = await build_shared_resources()
    s = SessionStore(res, PersistenceStore(tempfile.mktemp(suffix=".db")))
    assert s.get("does-not-exist") is None


def test_persistence_store_crud_and_purge():
    store = PersistenceStore(tempfile.mktemp(suffix=".db"))
    store.save_session("t1", "riya", time.time(), time.time())
    store.save_applications("t1", [{"company": "X"}, {"company": "Y"}])
    store.save_memory("t1", [{"role": "user", "content": "q"}])
    assert store.load_session("t1")["persona_id"] == "riya"
    assert len(store.load_applications("t1")) == 2
    assert store.load_memory("t1")[0]["content"] == "q"
    # expire everything
    store.save_session("t1", "riya", 0.0, 0.0)
    assert store.purge_expired(_SESSION_TTL_SECONDS) == 1
    assert store.load_session("t1") is None
    assert store.load_applications("t1") == []
