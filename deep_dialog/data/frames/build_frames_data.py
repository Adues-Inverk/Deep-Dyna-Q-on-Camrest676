"""
Extract KB, user goals, and slot dictionary from the Frames dataset.

Outputs (all in the same directory as this script):
  frames_kb.p            – dict[int, hotel_dict]
  frames_user_goals.p    – list[goal_dict]  (inform_slots + request_slots)
  frames_dict.p          – dict[slot, list[value]]  (valid values per slot)
  slot_set_frames.txt    – generated automatically
  dia_acts_frames.txt    – generated automatically
"""

import json
import os
import pickle
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_JSON = os.path.join(HERE, "..", "Frames-dataset", "Frames-dataset", "frames.json")

BUDGET_BINS = [("budget", 0, 1500), ("moderate", 1500, 2500), ("luxury", 2500, 1e9)]

def to_budget_range(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    for label, lo, hi in BUDGET_BINS:
        if lo <= v < hi:
            return label
    return "luxury"


def extract_hotels(data):
    """Return dict[int, hotel_dict] from wizard DB entries."""
    hotels = {}  # name -> attrs
    for dialog in data:
        for turn in dialog["turns"]:
            if turn["author"] != "wizard":
                continue
            db = turn.get("db", {})
            for frame_results in db.get("result", []):
                if not isinstance(frame_results, list):
                    continue
                for item in frame_results:
                    if not isinstance(item, dict) or "hotel" not in item:
                        continue
                    h = item["hotel"]
                    name = h.get("name", "")
                    if not name:
                        continue
                    amenities = ",".join(sorted(h.get("amenities", [])))
                    raw_price = item.get("price", 0.0) or 0.0
                    entry = {
                        "name":        name,
                        "dst_city":    h.get("dst_city", ""),
                        "category":    h.get("category", ""),
                        "gst_rating":  str(round(h.get("gst_rating", 0.0), 1)),
                        "amenities":   amenities,
                        "price":       str(round(raw_price, 2)),
                        "_price_num":  raw_price,
                    }
                    if name not in hotels:
                        hotels[name] = entry
                    else:
                        # keep lower price
                        if raw_price > 0 and (hotels[name]["_price_num"] == 0 or raw_price < hotels[name]["_price_num"]):
                            hotels[name]["price"] = str(round(raw_price, 2))
                            hotels[name]["_price_num"] = raw_price
    for h in hotels.values():
        p = h.pop("_price_num", 0.0)
        h["budget_range"] = to_budget_range(p) or "moderate"
    kb = {i: v for i, v in enumerate(hotels.values())}
    return kb


def extract_goals(data, kb):
    """Extract user goals from dialog first-turn user acts."""
    # Build lookup: city -> list of hotel ids
    city_hotels = defaultdict(list)
    for hid, h in kb.items():
        city_hotels[h["dst_city"]].append(hid)

    goals = []
    for dialog in data:
        inform_slots = {}
        for turn in dialog["turns"]:
            if turn["author"] != "user":
                continue
            for act in turn["labels"]["acts"]:
                for arg in act["args"]:
                    k, v = arg.get("key"), arg.get("val")
                    if k == "dst_city" and isinstance(v, str) and "dst_city" not in inform_slots:
                        inform_slots["dst_city"] = v
                    elif k == "budget" and "budget_range" not in inform_slots:
                        br = to_budget_range(v)
                        if br:
                            inform_slots["budget_range"] = br
                    elif k == "n_adults" and "n_adults" not in inform_slots:
                        try:
                            n = str(int(float(v)))
                            if 1 <= int(n) <= 8:
                                inform_slots["n_adults"] = n
                        except (TypeError, ValueError):
                            pass
            # Only parse the first user turn
            break

        if not inform_slots:
            continue

        # Skip goals where dst_city is not in the KB (fictional cities)
        if "dst_city" in inform_slots:
            kb_cities = {h["dst_city"] for h in kb.values()}
            if inform_slots["dst_city"] not in kb_cities:
                continue

        # Request slots: user always wants to know name, price, and gst_rating
        request_slots = {"name": "UNK", "price": "UNK", "gst_rating": "UNK"}

        goals.append({
            "diaact":       "request",
            "inform_slots": inform_slots,
            "request_slots": request_slots,
        })

    return goals


def build_slot_dict(kb):
    """Valid values per informable slot."""
    d = defaultdict(set)
    for h in kb.values():
        d["dst_city"].add(h["dst_city"])
        d["category"].add(h["category"])
        d["gst_rating"].add(str(h["gst_rating"]))
        d["amenities"].add(h["amenities"])
        d["name"].add(h["name"])
        d["price"].add(h["price"])
    d["budget_range"] = {"budget", "moderate", "luxury"}
    d["n_adults"]     = {str(i) for i in range(1, 9)}
    return {k: sorted(v) for k, v in d.items()}


def write_txt(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    print("Loading frames.json …")
    with open(FRAMES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} dialogs")

    kb = extract_hotels(data)
    print(f"  {len(kb)} unique hotels → frames_kb.p")
    with open(os.path.join(HERE, "frames_kb.p"), "wb") as f:
        pickle.dump(kb, f)

    goals = extract_goals(data, kb)
    print(f"  {len(goals)} user goals → frames_user_goals.p")
    with open(os.path.join(HERE, "frames_user_goals.p"), "wb") as f:
        pickle.dump(goals, f)

    slot_dict = build_slot_dict(kb)
    print(f"  slot_dict keys: {list(slot_dict.keys())} → frames_dict.p")
    with open(os.path.join(HERE, "frames_dict.p"), "wb") as f:
        pickle.dump(slot_dict, f)

    # Act set
    acts = ["request", "inform", "confirm_question", "confirm_answer", "thanks",
            "deny", "closing", "no_result", "sorry", "greeting", "moreinfo"]
    write_txt(os.path.join(HERE, "dia_acts_frames.txt"), acts)
    print(f"  dia_acts_frames.txt: {len(acts)} acts")

    # Slot set
    slots = ["dst_city", "budget_range", "n_adults", "category",
             "name", "price", "gst_rating", "amenities",
             "taskcomplete", "closing"]
    write_txt(os.path.join(HERE, "slot_set_frames.txt"), slots)
    print(f"  slot_set_frames.txt: {len(slots)} slots")

    print("Done.")


if __name__ == "__main__":
    main()
