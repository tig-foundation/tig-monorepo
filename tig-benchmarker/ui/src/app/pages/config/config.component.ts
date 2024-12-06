import { Component, inject, input, signal } from '@angular/core';

import { CodeEditorModule, CodeModel } from '@ngstack/code-editor';
import {
  FormControl,
  FormGroup,
  FormsModule,
  ReactiveFormsModule,
} from '@angular/forms';
import { InputTextModule } from 'primeng/inputtext';
import { CardModule } from 'primeng/card';
import { InputNumberModule } from 'primeng/inputnumber';
import { ButtonModule } from 'primeng/button';
import { merge, tap } from 'rxjs';
import { toSignal } from '@angular/core/rxjs-interop';
import { TigApisService } from '../../services/tig-apis.service';
import { SliderModule } from 'primeng/slider';
import { DropdownModule } from 'primeng/dropdown';
@Component({
  selector: 'app-settings',
  standalone: true,

  imports: [
    CardModule,
    ButtonModule,
    ReactiveFormsModule,
    FormsModule,
    CodeEditorModule,
    SliderModule,
    DropdownModule,
    InputNumberModule,
    InputTextModule,
  ],
  templateUrl: './config.component.html',
  styleUrl: './config.component.scss',
})
export class ConfigComponent {
  tigService: any = inject(TigApisService);
  settings: any = signal(null);
  slave: any = input<any>(null);
  error: any = signal(null);
  currentConfig = signal(null);
  theme = 'vs-dark';
  model: CodeModel = {
    language: 'json',
    uri: 'main.json',
    value: '{}',
  };
  options = {
    contextmenu: true,
    minimap: {
      enabled: true,
    },
  };
  async ngOnInit() {
    this.tigService.config$.subscribe((data: any) => {
      this.setCode(data);
    });
  }

  onCodeChanged(value: any) {
    this.currentConfig.set(value);
    this.testCode();
  }

  testCode() {
    // check valid json
    try {
      const code: any = this.currentConfig();
      JSON.parse(code);
      this.error.set(null);
    } catch (e) {
      this.error.set('Config is not valid JSON');
    }
  }

  setCode(code: any) {
    this.currentConfig.set(code);
    this.model = {
      language: 'json',
      uri: 'main.json',
      value: JSON.stringify(code, null, 2),
    };
  }

  async save() {
    this.tigService.saveConfig(this.currentConfig());
  }
}
