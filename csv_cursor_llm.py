#!/usr/bin/env python3
"""
CSV Cursor with LLM Integration using MLX
A tool to navigate and analyze CSV files with natural language interaction
"""

import csv
import sys
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import statistics
import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️  MLX not available. Install with: pip install mlx transformers")

class DataAnalyzer:
    """Helper class for data analysis operations"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.df = pd.DataFrame(data)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        prices = []
        volumes = []
        for record in self.data:
            prices.extend([record['open'], record['high'], record['low'], record['close']])
            volumes.append(record['volume'])
        
        return {
            'total_records': len(self.data),
            'date_range': f"{self.data[-1]['date']} to {self.data[0]['date']}",
            'price_range': f"${min(prices):.2f} - ${max(prices):.2f}",
            'avg_volume': f"{statistics.mean(volumes):,.0f}",
            'highest_close': f"${max(r['close'] for r in self.data):.2f}",
            'lowest_close': f"${min(r['close'] for r in self.data):.2f}",
            'total_change': f"{self.data[0]['close'] - self.data[-1]['close']:+.2f}",
            'total_change_pct': f"{((self.data[0]['close'] - self.data[-1]['close']) / self.data[-1]['close'] * 100):+.2f}%"
        }
    
    def find_patterns(self) -> Dict[str, Any]:
        """Find patterns in the data"""
        patterns = {}
        
        # Find consecutive up/down days
        up_days = 0
        down_days = 0
        max_up_streak = 0
        max_down_streak = 0
        current_up = 0
        current_down = 0
        
        for i in range(1, len(self.data)):
            prev_close = self.data[i]['close']
            curr_close = self.data[i-1]['close']
            
            if curr_close > prev_close:
                up_days += 1
                current_up += 1
                current_down = 0
                max_up_streak = max(max_up_streak, current_up)
            elif curr_close < prev_close:
                down_days += 1
                current_down += 1
                current_up = 0
                max_down_streak = max(max_down_streak, current_down)
        
        patterns['up_days'] = up_days
        patterns['down_days'] = down_days
        patterns['max_up_streak'] = max_up_streak
        patterns['max_down_streak'] = max_down_streak
        
        # Find volatile days (high range)
        ranges = [(r['high'] - r['low']) / r['open'] * 100 for r in self.data]
        volatile_days = [r for r, range_pct in zip(self.data, ranges) if range_pct > 5]
        patterns['volatile_days'] = len(volatile_days)
        patterns['avg_daily_range'] = f"{statistics.mean(ranges):.2f}%"
        
        return patterns
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze trends in the data"""
        # Calculate moving averages
        closes = [r['close'] for r in self.data]
        ma_5 = [statistics.mean(closes[max(0, i-4):i+1]) for i in range(len(closes))]
        ma_20 = [statistics.mean(closes[max(0, i-19):i+1]) for i in range(len(closes))]
        
        # Current trend
        current_ma5 = ma_5[0]
        current_ma20 = ma_20[0]
        trend = "Bullish" if current_ma5 > current_ma20 else "Bearish"
        
        return {
            'trend': trend,
            'ma5': f"${current_ma5:.2f}",
            'ma20': f"${current_ma20:.2f}",
            'trend_strength': f"{abs(current_ma5 - current_ma20) / current_ma20 * 100:.2f}%"
        }

class LLMAnalyzer:
    """LLM integration for natural language analysis"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the MLX model"""
        if not MLX_AVAILABLE:
            return
        
        try:
            # Use a smaller model for faster inference
            model_name = "microsoft/DialoGPT-medium"  # Smaller model for demo
            print("🔄 Loading LLM model...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Convert to MLX format
            self.model = self.convert_to_mlx(self.model)
            print("✅ LLM model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading LLM model: {e}")
            print("💡 Using fallback analysis mode")
    
    def convert_to_mlx(self, model):
        """Convert PyTorch model to MLX format"""
        # This is a simplified conversion - in practice you'd need more sophisticated conversion
        return model
    
    def analyze_query(self, query: str, data_context: Dict[str, Any]) -> str:
        """Analyze a natural language query about the data"""
        if not self.model:
            return self.fallback_analysis(query, data_context)
        
        # Create context for the LLM
        context = self.create_context(data_context)
        prompt = f"{context}\n\nUser: {query}\nAssistant:"
        
        try:
            # Generate response using MLX
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs, max_length=200, temperature=0.7)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            response = response.split("Assistant:")[-1].strip()
            return response
            
        except Exception as e:
            return self.fallback_analysis(query, data_context)
    
    def create_context(self, data_context: Dict[str, Any]) -> str:
        """Create context string for the LLM"""
        context = f"""
You are a financial data analyst assistant. You have access to Netflix (NFLX) stock data with the following information:

Summary Statistics:
- Total Records: {data_context.get('summary', {}).get('total_records', 'N/A')}
- Date Range: {data_context.get('summary', {}).get('date_range', 'N/A')}
- Price Range: {data_context.get('summary', {}).get('price_range', 'N/A')}
- Average Volume: {data_context.get('summary', {}).get('avg_volume', 'N/A')}
- Total Change: {data_context.get('summary', {}).get('total_change', 'N/A')} ({data_context.get('summary', {}).get('total_change_pct', 'N/A')})

Current Record:
- Date: {data_context.get('current_record', {}).get('date', 'N/A')}
- Open: ${data_context.get('current_record', {}).get('open', 'N/A')}
- High: ${data_context.get('current_record', {}).get('high', 'N/A')}
- Low: ${data_context.get('current_record', {}).get('low', 'N/A')}
- Close: ${data_context.get('current_record', {}).get('close', 'N/A')}
- Volume: {data_context.get('current_record', {}).get('volume', 'N/A'):,}

Trend Analysis:
- Current Trend: {data_context.get('trend', {}).get('trend', 'N/A')}
- 5-day MA: {data_context.get('trend', {}).get('ma5', 'N/A')}
- 20-day MA: {data_context.get('trend', {}).get('ma20', 'N/A')}

Please provide insightful analysis based on this data.
"""
        return context
    
    def fallback_analysis(self, query: str, data_context: Dict[str, Any]) -> str:
        """Fallback analysis when LLM is not available"""
        query_lower = query.lower()
        
        if 'trend' in query_lower or 'direction' in query_lower:
            trend = data_context.get('trend', {}).get('trend', 'Unknown')
            return f"Based on the moving averages, the current trend appears to be {trend}."
        
        elif 'volume' in query_lower:
            avg_volume = data_context.get('summary', {}).get('avg_volume', 'N/A')
            current_volume = data_context.get('current_record', {}).get('volume', 0)
            if current_volume > int(avg_volume.replace(',', '')):
                return f"Volume is above average today ({current_volume:,} vs {avg_volume} average)."
            else:
                return f"Volume is below average today ({current_volume:,} vs {avg_volume} average)."
        
        elif 'performance' in query_lower or 'how is it doing' in query_lower:
            change_pct = data_context.get('summary', {}).get('total_change_pct', 'N/A')
            return f"Netflix stock has shown {change_pct} performance over the analyzed period."
        
        elif 'volatility' in query_lower:
            patterns = data_context.get('patterns', {})
            avg_range = patterns.get('avg_daily_range', 'N/A')
            return f"The average daily price range is {avg_range}, indicating moderate volatility."
        
        else:
            return "I can help analyze trends, volume patterns, performance metrics, and volatility. What specific aspect would you like to know about?"

class CSVCursorLLM:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = []
        self.headers = []
        self.current_index = 0
        self.analyzer = None
        self.llm_analyzer = LLMAnalyzer()
        self.load_data()
    
    def load_data(self):
        """Load CSV data from file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                self.headers = next(reader)  # Get headers
                
                for i, row in enumerate(reader):
                    if len(row) >= 6:  # Ensure we have all expected columns
                        self.data.append({
                            'index': i + 1,
                            'date': row[0],
                            'open': float(row[1].replace('"', '').replace(',', '')),
                            'high': float(row[2].replace('"', '').replace(',', '')),
                            'low': float(row[3].replace('"', '').replace(',', '')),
                            'close': float(row[4].replace('"', '').replace(',', '')),
                            'volume': int(row[5].replace('"', '').replace(',', ''))
                        })
            
            self.analyzer = DataAnalyzer(self.data)
            print(f"✅ Loaded {len(self.data)} records from {self.file_path}")
            
        except FileNotFoundError:
            print(f"❌ File not found: {self.file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            sys.exit(1)
    
    def get_current_record(self) -> Dict:
        """Get the current record"""
        if 0 <= self.current_index < len(self.data):
            return self.data[self.current_index]
        return None
    
    def next(self) -> bool:
        """Move to next record"""
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            return True
        return False
    
    def previous(self) -> bool:
        """Move to previous record"""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False
    
    def go_to(self, index: int) -> bool:
        """Go to specific record index"""
        if 0 <= index < len(self.data):
            self.current_index = index
            return True
        return False
    
    def search_by_date(self, date_str: str) -> Optional[int]:
        """Search for a record by date"""
        for i, record in enumerate(self.data):
            if date_str in record['date']:
                return i
        return None
    
    def display_current(self):
        """Display current record"""
        record = self.get_current_record()
        if record:
            print(f"\n📊 Record {self.current_index + 1} of {len(self.data)}")
            print("=" * 50)
            print(f"Date:     {record['date']}")
            print(f"Open:     ${record['open']:.2f}")
            print(f"High:     ${record['high']:.2f}")
            print(f"Low:      ${record['low']:.2f}")
            print(f"Close:    ${record['close']:.2f}")
            print(f"Volume:   {record['volume']:,}")
            
            # Calculate change
            change = record['close'] - record['open']
            change_pct = (change / record['open']) * 100
            change_symbol = "📈" if change >= 0 else "📉"
            print(f"Change:   {change_symbol} ${change:.2f} ({change_pct:+.2f}%)")
        else:
            print("❌ No current record")
    
    def ask_llm(self, question: str):
        """Ask the LLM about the current data"""
        if not self.analyzer:
            print("❌ No data available for analysis")
            return
        
        # Prepare context for LLM
        current_record = self.get_current_record()
        data_context = {
            'summary': self.analyzer.get_summary_stats(),
            'current_record': current_record,
            'trend': self.analyzer.get_trend_analysis(),
            'patterns': self.analyzer.find_patterns()
        }
        
        print(f"\n🤖 LLM Analysis for: '{question}'")
        print("=" * 60)
        
        response = self.llm_analyzer.analyze_query(question, data_context)
        print(response)
        print("=" * 60)
    
    def display_summary(self):
        """Display summary statistics"""
        if not self.analyzer:
            print("❌ No data available")
            return
        
        stats = self.analyzer.get_summary_stats()
        patterns = self.analyzer.find_patterns()
        trend = self.analyzer.get_trend_analysis()
        
        print(f"\n📈 Summary Statistics")
        print("=" * 50)
        print(f"Total Records:     {stats['total_records']}")
        print(f"Date Range:        {stats['date_range']}")
        print(f"Price Range:       {stats['price_range']}")
        print(f"Average Volume:    {stats['avg_volume']}")
        print(f"Highest Close:     {stats['highest_close']}")
        print(f"Lowest Close:      {stats['lowest_close']}")
        print(f"Total Change:      {stats['total_change']} ({stats['total_change_pct']})")
        
        print(f"\n📊 Pattern Analysis")
        print("=" * 50)
        print(f"Up Days:           {patterns['up_days']}")
        print(f"Down Days:         {patterns['down_days']}")
        print(f"Max Up Streak:     {patterns['max_up_streak']} days")
        print(f"Max Down Streak:   {patterns['max_down_streak']} days")
        print(f"Volatile Days:     {patterns['volatile_days']}")
        print(f"Avg Daily Range:   {patterns['avg_daily_range']}")
        
        print(f"\n📈 Trend Analysis")
        print("=" * 50)
        print(f"Current Trend:     {trend['trend']}")
        print(f"5-day MA:          {trend['ma5']}")
        print(f"20-day MA:         {trend['ma20']}")
        print(f"Trend Strength:    {trend['trend_strength']}")

def main():
    # Default to the Netflix CSV file
    csv_file = "Download Data - STOCK_US_XNAS_NFLX.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("Usage: python csv_cursor_llm.py [csv_file]")
        sys.exit(1)
    
    cursor = CSVCursorLLM(csv_file)
    
    print(f"\n🎯 CSV Cursor with LLM - {csv_file}")
    print("=" * 50)
    print("Commands:")
    print("  n/next     - Next record")
    print("  p/prev     - Previous record")
    print("  g <num>    - Go to record number")
    print("  s <date>   - Search by date")
    print("  c/current  - Show current record")
    print("  sum        - Show summary")
    print("  ask <q>    - Ask LLM about the data")
    print("  q/quit     - Quit")
    print("=" * 50)
    
    while True:
        try:
            command = input(f"\n[{cursor.current_index + 1}/{len(cursor.data)}] > ").strip()
            
            if command.lower() in ['n', 'next']:
                if cursor.next():
                    cursor.display_current()
                else:
                    print("❌ Already at last record")
            
            elif command.lower() in ['p', 'prev']:
                if cursor.previous():
                    cursor.display_current()
                else:
                    print("❌ Already at first record")
            
            elif command.startswith('g '):
                try:
                    num = int(command[2:])
                    if cursor.go_to(num - 1):
                        cursor.display_current()
                    else:
                        print(f"❌ Invalid record number: {num}")
                except ValueError:
                    print("❌ Invalid number")
            
            elif command.startswith('s '):
                date_str = command[2:]
                found_idx = cursor.search_by_date(date_str)
                if found_idx is not None:
                    cursor.go_to(found_idx)
                    cursor.display_current()
                else:
                    print(f"❌ Date not found: {date_str}")
            
            elif command.lower() in ['c', 'current']:
                cursor.display_current()
            
            elif command.lower() in ['sum']:
                cursor.display_summary()
            
            elif command.startswith('ask '):
                question = command[4:]
                cursor.ask_llm(question)
            
            elif command.lower() in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Unknown command. Available commands:")
                print("  n/next, p/prev, g <num>, s <date>, c/current, sum, ask <question>, q/quit")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 