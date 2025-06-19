#!/usr/bin/env python3
"""
Setup script to download Cricket World Cup 2015 data and create DuckDB database.
"""

import duckdb
import pandas as pd
import requests
import os
from pathlib import Path


def download_csv_files():
    """Download all CSV files from the Cricket WC 2015 repository."""
    base_url = "https://raw.githubusercontent.com/rishsriv/cricket-wc2015/master/wc_cleaned_data/"

    # Create data directory
    data_dir = Path("cricket_data")
    data_dir.mkdir(exist_ok=True)

    # List of match IDs (extracted from the repository)
    match_ids = [
        656399,
        656401,
        656403,
        656405,
        656407,
        656409,
        656411,
        656413,
        656415,
        656417,
        656419,
        656421,
        656423,
        656425,
        656427,
        656429,
        656431,
        656433,
        656435,
        656437,
        656439,
        656441,
        656443,
        656445,
        656447,
        656449,
        656451,
        656453,
        656455,
        656457,
        656459,
        656461,
        656463,
        656465,
        656467,
        656469,
        656471,
        656473,
        656475,
        656477,
        656479,
        656481,
        656483,
        656485,
        656487,
        656489,
        656491,
        656493,
        656495,
    ]

    downloaded_files = []

    for match_id in match_ids:
        filename = f"{match_id}_summary.csv"
        url = base_url + filename
        filepath = data_dir / filename

        # Check if file already exists
        if filepath.exists():
            print(f"✓ {filename} already exists, skipping download")
            downloaded_files.append(filepath)
            continue

        try:
            print(f"Downloading {filename}...")
            response = requests.get(url)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)

            downloaded_files.append(filepath)
            print(f"✓ Downloaded {filename}")

        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")

    return downloaded_files


def create_cricket_database(csv_files, db_path="cricket_wc2015.duckdb"):
    """Create DuckDB database from CSV files."""
    print(f"\nCreating DuckDB database: {db_path}")

    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    # First, let's check the structure of the CSV files
    sample_df = pd.read_csv(csv_files[0], index_col=0)
    print(f"CSV columns: {list(sample_df.columns)}")

    # Check what columns are available in the first CSV
    conn.execute(
        f"CREATE TEMP TABLE sample_csv AS SELECT * FROM read_csv('{csv_files[0]}', auto_detect=true)"
    )
    columns_result = conn.execute("PRAGMA table_info('sample_csv')").fetchall()
    conn.execute("DROP TABLE sample_csv")
    csv_columns = [col[1] for col in columns_result]
    print(f"Available CSV columns: {csv_columns}")

    # Create main table for ball-by-ball data with flexible schema
    conn.execute(
        """
        CREATE TABLE ball_by_ball (
            match_id INTEGER,
            inning INTEGER,
            batting_team TEXT,
            bowling_team TEXT,
            batsman TEXT,
            bowler TEXT,
            batsman_name TEXT,
            non_striker TEXT,
            bowler_name TEXT,
            bat_right_handed TEXT,
            ovr REAL,
            runs_batter REAL,
            runs_w_extras REAL,
            extras REAL,
            cumul_runs REAL,
            wicket TEXT,
            wicket_method TEXT,
            who_out TEXT,
            control INTEGER,
            extras_type TEXT,
            x REAL,
            y REAL,
            z REAL,
            landing_x REAL,
            landing_y REAL,
            ended_x REAL,
            ended_y REAL,
            ball_speed REAL
        )
    """
    )

    # Process each CSV file
    total_records = 0
    matches_processed = 0

    for csv_file in csv_files:
        try:
            print(f"Processing {csv_file.name}...")

            # Extract match ID from filename
            match_id = int(csv_file.stem.split("_")[0])

            # Use DuckDB to read CSV directly and insert
            temp_table_name = f"temp_match_{match_id}"

            # Read CSV into temporary table
            conn.execute(
                f"""
                CREATE TEMP TABLE {temp_table_name} AS 
                SELECT *, {match_id} as match_id 
                FROM read_csv('{csv_file}', auto_detect=true)
            """
            )

            # Insert into main table with proper column ordering
            conn.execute(
                f"""
                INSERT INTO ball_by_ball 
                SELECT 
                    match_id,
                    inning,
                    batting_team,
                    bowling_team,
                    batsman,
                    bowler,
                    batsman_name,
                    non_striker,
                    bowler_name,
                    bat_right_handed,
                    ovr,
                    runs_batter,
                    runs_w_extras,
                    extras,
                    cumul_runs,
                    wicket,
                    wicket_method,
                    who_out,
                    control,
                    extras_type,
                    x,
                    y,
                    z,
                    landing_x,
                    landing_y,
                    ended_x,
                    ended_y,
                    ball_speed
                FROM {temp_table_name}
            """
            )

            # Get record count and clean up
            record_count = conn.execute(
                f"SELECT COUNT(*) FROM {temp_table_name}"
            ).fetchone()[0]
            conn.execute(f"DROP TABLE {temp_table_name}")

            total_records += record_count
            matches_processed += 1
            print(f"✓ Processed {csv_file.name} - {record_count} records")

        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {e}")

    # Create indexes for better query performance
    print("\nCreating database indexes...")
    conn.execute("CREATE INDEX idx_match_id ON ball_by_ball(match_id)")
    conn.execute("CREATE INDEX idx_batsman ON ball_by_ball(batsman)")
    conn.execute("CREATE INDEX idx_bowler ON ball_by_ball(bowler)")
    conn.execute("CREATE INDEX idx_batting_team ON ball_by_ball(batting_team)")
    conn.execute("CREATE INDEX idx_bowling_team ON ball_by_ball(bowling_team)")

    # Create summary tables for easier querying
    print("Creating summary tables...")

    # Match summary
    conn.execute(
        """
        CREATE TABLE match_summary AS
        SELECT 
            match_id,
            COUNT(*) as total_balls,
            SUM(runs_w_extras) as total_runs,
            COUNT(DISTINCT batting_team) as teams,
            MIN(batting_team) as team1,
            MAX(batting_team) as team2
        FROM ball_by_ball 
        GROUP BY match_id
    """
    )

    # Team performance summary
    conn.execute(
        """
        CREATE TABLE team_performance AS
        SELECT 
            batting_team as team,
            COUNT(*) as balls_faced,
            SUM(runs_batter) as runs_scored,
            SUM(CASE WHEN wicket = '1' THEN 1 ELSE 0 END) as wickets_lost,
            COUNT(DISTINCT match_id) as matches_played
        FROM ball_by_ball 
        GROUP BY batting_team
    """
    )

    # Player batting stats
    conn.execute(
        """
        CREATE TABLE player_batting_stats AS
        SELECT 
            batsman_name,
            COUNT(*) as balls_faced,
            SUM(runs_batter) as runs_scored,
            SUM(CASE WHEN runs_batter = 4 THEN 1 ELSE 0 END) as fours,
            SUM(CASE WHEN runs_batter = 6 THEN 1 ELSE 0 END) as sixes,
            SUM(CASE WHEN wicket = '1' AND CAST(who_out AS TEXT) = batsman THEN 1 ELSE 0 END) as dismissals,
            COUNT(DISTINCT match_id) as matches_played
        FROM ball_by_ball 
        WHERE batsman_name IS NOT NULL AND batsman_name != ''
        GROUP BY batsman_name
    """
    )

    # Player bowling stats
    conn.execute(
        """
        CREATE TABLE player_bowling_stats AS
        SELECT 
            bowler_name,
            COUNT(*) as balls_bowled,
            SUM(runs_w_extras) as runs_conceded,
            SUM(CASE WHEN wicket = '1' AND wicket_method NOT IN ('run out', 'retired hurt') THEN 1 ELSE 0 END) as wickets_taken,
            COUNT(DISTINCT match_id) as matches_played
        FROM ball_by_ball 
        WHERE bowler_name IS NOT NULL AND bowler_name != ''
        GROUP BY bowler_name
    """
    )

    conn.commit()
    conn.close()

    print(f"\n✓ Database created successfully!")
    print(f"  - Total records: {total_records}")
    print(f"  - Matches processed: {matches_processed}")
    print(f"  - Database file: {db_path}")

    return db_path


def main():
    """Main function to set up Cricket World Cup 2015 database."""
    print("Setting up Cricket World Cup 2015 Database")
    print("=" * 50)

    # Download CSV files
    csv_files = download_csv_files()

    if not csv_files:
        print("No CSV files downloaded. Exiting.")
        return

    # Create database
    db_path = create_cricket_database(csv_files)

    # Test the database
    print("\nTesting database...")
    conn = duckdb.connect(db_path)

    # Get some basic stats
    total_balls = conn.execute("SELECT COUNT(*) FROM ball_by_ball").fetchone()[0]
    total_matches = conn.execute(
        "SELECT COUNT(DISTINCT match_id) FROM ball_by_ball"
    ).fetchone()[0]
    unique_batsmen = conn.execute(
        "SELECT COUNT(DISTINCT batsman_name) FROM ball_by_ball WHERE batsman_name IS NOT NULL"
    ).fetchone()[0]
    unique_bowlers = conn.execute(
        "SELECT COUNT(DISTINCT bowler_name) FROM ball_by_ball WHERE bowler_name IS NOT NULL"
    ).fetchone()[0]

    print(f"Database Statistics:")
    print(f"  Total balls: {total_balls}")
    print(f"  Total matches: {total_matches}")
    print(f"  Unique batsmen: {unique_batsmen}")
    print(f"  Unique bowlers: {unique_bowlers}")

    conn.close()

    print(f"\n✓ Setup complete! Database ready at: {os.path.abspath(db_path)}")


if __name__ == "__main__":
    main()
