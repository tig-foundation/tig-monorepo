import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { NavBarComponent } from "./components/nav-bar/nav-bar.component";
import { SidebarModule } from 'primeng/sidebar';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ButtonModule } from 'primeng/button';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, NavBarComponent,SidebarModule,FormsModule,ReactiveFormsModule,ButtonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  sidebarVisible=true;
  title = 'tig-brenchmarker-ui';
}
