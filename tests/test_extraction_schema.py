from __future__ import annotations

import pytest

from app.extraction import SchemaError, normalize_schema

RAW = {
    "name": "thing",
    "description": "a thing to extract",
    "fields": {
        "alpha": {"name": "Alpha", "description": "the a field"},
        "beta": {"name": "Beta"},
        "gamma": {"description": "the g field", "primary": True},
    },
}


def test_preserves_keys_names_descriptions_primary():
    schema = normalize_schema(RAW)
    assert list(schema.fields) == ["alpha", "beta", "gamma"]
    assert schema.fields["alpha"].name == "Alpha"
    assert schema.fields["alpha"].description == "the a field"
    assert schema.fields["beta"].name == "Beta"
    assert schema.fields["beta"].description == ""
    assert schema.fields["beta"].primary is False
    assert schema.fields["gamma"].primary is True
    assert schema.primary_key == "gamma"
    assert schema.name == "thing"
    assert schema.description == "a thing to extract"


def test_missing_top_level_name_and_description_default_empty():
    schema = normalize_schema({"fields": {"x": {"name": "X"}}})
    assert schema.name == ""
    assert schema.description == ""


def test_primary_flag_false_explicit():
    schema = normalize_schema({"fields": {"x": {"name": "X", "primary": False}}})
    assert schema.primary_key is None


def test_no_primary_is_fine():
    schema = normalize_schema({"fields": {"x": {"name": "X"}}})
    assert schema.primary_key is None


@pytest.mark.parametrize(
    "bad",
    [
        [],
        "schema",
        {},
        {"name": "n"},
        {"fields": {}},
        {"fields": []},
        {"fields": {"a": 5}},
        {"fields": {"a": {"name": 1}}},
        {"fields": {"a": {"description": 3}}},
        {"fields": {"a": {"primary": "yes"}}},
        {"fields": {"a": {"name": "A", "primary": True}, "b": {"name": "B", "primary": True}}},
        {"fields": {"a": {"name": "A"}}, "name": 7},
        {"fields": {"a": {"name": "A"}}, "description": 7},
    ],
)
def test_invalid_schemas_raise(bad):
    with pytest.raises(SchemaError):
        normalize_schema(bad)


def test_error_message_mentions_offending_field():
    with pytest.raises(SchemaError, match="'a'"):
        normalize_schema({"fields": {"a": {"name": 1}}})


def test_sample_contract_schema_normalizes():
    import json
    from pathlib import Path

    raw = json.loads(Path("contract/schema.example.json").read_text())
    schema = normalize_schema(raw)
    assert schema.primary_key == "code"
    assert list(schema.fields)[0] == "property_name"


def test_sample_schemas_all_normalize():
    import json
    from pathlib import Path

    for path in sorted(Path("sample/schemas").glob("*.json")):
        schema = normalize_schema(json.loads(path.read_text()))
        assert len(schema.fields) > 0
        assert schema.name == path.stem


def test_default_parsing_boolean_false_otherwise():
    schema = normalize_schema({"fields": {
        "loc": {"name": "LOC", "description": "true if the lease requires a letter of credit as security, false otherwise."},
    }})
    assert schema.fields["loc"].has_default is True
    assert schema.fields["loc"].default_value is False


def test_default_parsing_numeric_zero_if_none():
    schema = normalize_schema({"fields": {
        "count": {"name": "Count", "description": "How many renewal options the tenant holds, as a number. 0 if none."},
    }})
    assert schema.fields["count"].has_default is True
    assert schema.fields["count"].default_value == 0


def test_default_parsing_empty_list_if_none():
    schema = normalize_schema({"fields": {
        "brokers": {"name": "Brokers", "description": "List of the brokers named in the lease. Empty list if none are named."},
    }})
    assert schema.fields["brokers"].has_default is True
    assert schema.fields["brokers"].default_value == []


def test_default_parsing_null_if_is_not_a_default():
    schema = normalize_schema({"fields": {
        "guarantor": {"name": "Guarantor", "description": "Full name of any guarantor of the lease. null if there is no guaranty."},
        "late_fee": {"name": "Late Fee", "description": "The late payment charge as the lease states it. null if not stated."},
    }})
    assert schema.fields["guarantor"].has_default is False
    assert schema.fields["guarantor"].default_value is None
    assert schema.fields["late_fee"].has_default is False


def test_no_default_when_description_is_silent():
    schema = normalize_schema({"fields": {
        "name": {"name": "Name", "description": "The tenant name."},
    }})
    assert schema.fields["name"].has_default is False


def test_default_priority_boolean_wins_over_zero():
    schema = normalize_schema({"fields": {
        "flag": {"name": "Flag", "description": "false otherwise, 0 if none."},
    }})
    assert schema.fields["flag"].default_value is False


def test_default_parsing_boolean_false_when():
    schema = normalize_schema({"fields": {
        "flag": {"name": "Flag", "description": "Returns false when the tenant holds no option."},
    }})
    assert schema.fields["flag"].has_default is True
    assert schema.fields["flag"].default_value is False


def test_default_parsing_zero_negative_lookalikes_do_not_match():
    schema = normalize_schema({"fields": {
        "a": {"name": "A", "description": "Values range from 10 if the period is short."},
        "b": {"name": "B", "description": "Computed as 0.5 if partially vacant."},
    }})
    assert schema.fields["a"].has_default is False
    assert schema.fields["b"].has_default is False
