import sqlite3
import streamlit as st
from difflib import SequenceMatcher

@st.cache_resource
def get_database_connection():
    conn = sqlite3.connect('woodworking_buddy.db', check_same_thread=False)
    return conn

def init_db(conn):
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS products 
                 (id INTEGER PRIMARY KEY, 
                  title TEXT, 
                  tags TEXT, 
                  links TEXT)''')
    conn.commit()

def load_initial_data():
    try:
        conn = get_database_connection()
        data = [
            (1, "TSO Products", "Aftermarket Festool accessories, Precision woodworking tools, Router table inserts, Guide rail accessories, Dust collection adapters,TSO Products", "https://tsoproducts.com/?aff=5"),
            (2, "Bits and Bits Company", "Router bits, Drill bits, Saw blades, Woodworking accessories, Carbide cutting tools, Bits and Bits Company", "http://bit.ly/bitsbitsbw"),
            (3, "Taylor Toolworks", "Taylor Toolworks, Woodworking hand tools, Japanese saws, Chisels, Sharpening supplies, Layout tools", "https://lddy.no/1e5hv"),
            (4, "Festool LR 32 System", "Festool LR 32 System, Cabinet making, Shelf pin holes, Precision drilling, 32mm system, Modular shelving, European cabinetry, Drawer slide installation, Cabinet hardware installation", "https://amzn.to/3hRTvLB"),
            (5, "Festool Trigger Clamp", "Festool Trigger Clamp, Quick release, One-handed operation, Versatile clamping, Woodworking, Assembly, Glue-ups", "https://amzn.to/2HoVydC"),
            (6, "Festool LR 32 Rail", "Festool LR 32 Rail, Guide rail, 32mm hole spacing, Cabinet making, Shelf pin holes, Precision drilling, Modular, Aluminum extrusion", "https://amzn.to/33LsnsG"),
            (7, "Festool OF 1400", "Festool OF 1400, Plunge router, Variable speed, Dust extraction, Precision routing, Cabinet making, Edge profiling, Mortising", "https://amzn.to/2FRerp5"),
            (8, "Festool Vac Sys Head", "Festool Vac Sys Head, Vacuum clamping, Workholding system, Precision woodworking, Dust extraction, Versatile clamping", "https://amzn.to/3010rjw"),
            (9, "Festool Midi Vac", "Festool Midi Vac, Compact dust extractor, HEPA filtration, Auto-start, Variable suction, Systainer compatibility", "https://amzn.to/2HfMmrM"),
            (10, "Festool Bluetooth Switch", "Festool Bluetooth Switch, Remote dust extractor control, Wireless operation, Tool-triggered activation, Energy efficiency, Workshop convenience", "https://amzn.to/33RAt36"),
            (11, "Woodpeckers TS600", "Woodpeckers TS600, T-square, Precision measurement, Layout tool, Anodized aluminum, Woodworking, Imperial/metric scales", "https://amzn.to/3mIc34t")
        ]
        c = conn.cursor()
        c.executemany("INSERT OR REPLACE INTO products (id, title, tags, links) VALUES (?, ?, ?, ?)", data)
        conn.commit()
    except Exception as e:
        st.error(f"Error loading initial data: {str(e)}")

def get_all_products():
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("SELECT id, title, tags, links FROM products WHERE tags != '' AND links != ''")
    return c.fetchall()

def query_db_for_keywords(keywords):
    conn = get_database_connection()
    c = conn.cursor()
    
    c.execute("SELECT id, title, tags, links FROM products")
    all_products = c.fetchall()
    
    scored_results = []
    for product in all_products:
        score = calculate_score(keywords, product[2])
        if score > 0:
            scored_results.append((score, product))
    
    scored_results.sort(reverse=True, key=lambda x: x[0])
    return [result[1] for result in scored_results[:5]] if scored_results else None  # Return top 5 matches

def calculate_score(keywords, tags):
    tag_set = set(tag.lower().strip() for tag in tags.split(','))
    score = 0
    for keyword in keywords:
        keyword = keyword.lower().strip()
        best_match = max(tag_set, key=lambda tag: SequenceMatcher(None, keyword, tag).ratio())
        match_ratio = SequenceMatcher(None, keyword, best_match).ratio()
        if match_ratio > 0.7:  # You can adjust this threshold
            score += match_ratio
    return score

def add_product(title, tags, link):
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("INSERT INTO products (title, tags, links) VALUES (?, ?, ?)", (title, tags, link))
    conn.commit()

def delete_product(product_id):
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("DELETE FROM products WHERE id = ?", (product_id,))
    conn.commit()

def update_product(product_id, title, tags, link):
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("UPDATE products SET title = ?, tags = ?, links = ? WHERE id = ?", (title, tags, link, product_id))
    conn.commit()

def get_product_by_id(product_id):
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("SELECT id, title, tags, links FROM products WHERE id = ?", (product_id,))
    return c.fetchone()